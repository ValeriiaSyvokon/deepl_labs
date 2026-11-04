from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "seg_train" / "seg_train"
TEST_DIR = ROOT / "seg_test" / "seg_test"
SPLIT_FILE = ROOT / "split.csv"

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(CLASSES)
}

IMG_SIZE = 160
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 64

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms():
    """Створює трансформації для train та validation/test."""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.7, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_transform, eval_transform


class IntelDataset(Dataset):
    def __init__(self, part, transform=None):
        self.part = part
        self.transform = transform
        self.samples = []

        if part not in {"train", "validation", "test"}:
            raise ValueError(f"Невідома частина вибірки: {part}")

        with SPLIT_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row["part"] == part:
                    self.samples.append(row)

        if not self.samples:
            raise ValueError(f"Вибірка {part} порожня.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]

        class_name = row["label"]
        filename = row["file"]

        if class_name not in CLASS_TO_IDX:
            raise ValueError(f"Невідомий клас: {class_name}")

        if self.part == "train":
            image_path = TRAIN_DIR / class_name / filename
        else:
            image_path = TEST_DIR / class_name / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = CLASS_TO_IDX[class_name]

        return image, label, filename


def create_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = IntelDataset(
        part="train",
        transform=train_transform,
    )

    val_dataset = IntelDataset(
        part="validation",
        transform=eval_transform,
    )

    test_dataset = IntelDataset(
        part="test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def visualize_train_batch(train_loader, output_path=None):
    """Візуалізує перші 16 зображень першого train batch."""

    import matplotlib.pyplot as plt

    images, labels, filenames = next(iter(train_loader))

    images = images[:16]
    labels = labels[:16]
    filenames = filenames[:16]

    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)

    # Скасовуємо нормалізацію для відображення
    images = images * std + mean
    images = images.clamp(0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        image = images[i].permute(1, 2, 0).numpy()

        label_name = CLASSES[labels[i].item()]

        ax.imshow(image)
        ax.set_title(f"{label_name}\n{filenames[i]}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    images, labels, filenames = next(iter(train_loader))

    print("\nПерший train batch:")
    print("Форма зображень:", images.shape)
    print("Форма міток:", labels.shape)
    print("Кількість файлів:", len(filenames))

    print("\nПеревірка validation batch:")
    val_images, val_labels, val_filenames = next(iter(val_loader))
    print("Форма зображень:", val_images.shape)
    print("Форма міток:", val_labels.shape)

    visualize_train_batch(
        train_loader,
        ROOT / "figures" / "train_batch.png",
    )