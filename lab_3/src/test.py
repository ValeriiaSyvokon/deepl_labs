
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from torchvision.models import resnet18
from sklearn.metrics import f1_score, confusion_matrix

from dataset import create_dataloaders, CLASSES, IMG_SIZE


def evaluate_test(checkpoint_path):
    """
    Одноразова оцінка вибраної моделі на test.
    Викликається з analysis.py після перевірки вибору моделі.
    Обчислення виконуються на CPU.
    """

    device = torch.device("cpu")
    checkpoint_path = Path(checkpoint_path)

    # Завантаження фінальної моделі
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Перевірка метаданих
    assert checkpoint["regime"] == "unfreeze"
    assert checkpoint["seed"] == 2
    assert list(checkpoint["classes"]) == list(CLASSES)
    assert checkpoint["img_size"] == IMG_SIZE == 160

    # Відтворення архітектури ResNet-18
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    # Завантажувачі даних
    _, _, test_loader = create_dataloaders()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_filenames = []

    # Оцінювання
    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())
            all_probabilities.extend(probabilities.numpy())
            all_filenames.extend(filenames)

    y_true = np.array(all_labels)
    if len(y_true) != 1502:
        raise ValueError(
            f"Очікувалося 1502 тестових зображення, отримано {len(y_true)}."
        )
    y_pred = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    # Метрики
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0
    )

    accuracy = np.mean(y_true == y_pred)
    n = len(y_true)

    standard_error = np.sqrt(
        accuracy * (1 - accuracy) / n
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    print("\nОЦІНЮВАННЯ НА TEST")
    print("=" * 45)
    print(f"Пристрій: {device}")
    print(f"Кількість зображень: {n}")
    print(f"Test macro-F1: {macro_f1:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Стандартна похибка accuracy: {standard_error:.6f}")

    print("\nМатриця плутанини:")
    print(matrix)

    print("\nПорядок класів:")
    for i, name in enumerate(CLASSES):
        print(f"{i}: {name}")

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "standard_error": standard_error,
        "confusion_matrix": matrix,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "filenames": all_filenames,
    }
