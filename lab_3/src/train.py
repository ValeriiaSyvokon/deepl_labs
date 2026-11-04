import hashlib
import json
import subprocess
import time
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import torchvision

from sklearn.metrics import f1_score

from dataset import (
    create_dataloaders,
    CLASSES,
    IMG_SIZE,
    MEAN,
    STD,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_EVAL,
)
from model import (
    build_model,
    set_train_mode,
    get_device,
)


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
SPLIT_FILE = ROOT / "split.csv"

EPOCHS = 8

RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def sha256_file(path):
    """Обчислює SHA256 файлу."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_info():
    """Отримує commit hash і стан робочого дерева Git."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        return {
            "commit_hash": commit,
            "working_tree_dirty": bool(status),
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit_hash": None,
            "working_tree_dirty": None,
        }


def device_info(device):
    """Повертає назву пристрою."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)

    if device.type == "mps":
        return "Apple MPS"

    return platform.processor() or "CPU"


def evaluate(model, loader, device):
    """Обчислює validation macro-F1 та повертає logits."""
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)

            logits = model(images)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    predictions = np.argmax(logits, axis=1)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return macro_f1, logits


def train_one_epoch(model, optimizer, loader, device, mode):
    """Навчає модель протягом однієї епохи."""
    set_train_mode(model, mode)

    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def run_experiment(mode, seed, train_loader, val_loader, device):
    """Виконує один повний запуск моделі."""
    run_name = f"{mode}_seed{seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ЗАПУСК: {run_name}")
    print("=" * 70)

    model, optimizer = build_model(
        mode=mode,
        seed=seed,
        device=device,
    )

    train_losses = []
    validation_f1 = []
    epoch_seconds = []

    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            mode,
        )

        val_f1, _ = evaluate(
            model,
            val_loader,
            device,
        )

        elapsed = time.perf_counter() - epoch_start

        train_losses.append(float(train_loss))
        validation_f1.append(float(val_f1))
        epoch_seconds.append(float(elapsed))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val macro-F1: {val_f1:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

    training_seconds = time.perf_counter() - total_start

    # Після восьмої епохи зберігаємо validation logits
    final_f1, val_logits = evaluate(
        model,
        val_loader,
        device,
    )

    np.save(run_dir / "val_logits.npy", val_logits)

    # Конфігурація запуску
    config = {
        "mode": mode,
        "seed": seed,

        "model": {
            "architecture": "ResNet-18",
            "weights": (
                None if mode == "scratch"
                else "ResNet18_Weights.IMAGENET1K_V1"
            ),
            "num_classes": len(CLASSES),
            "classes": CLASSES,
            "img_size": IMG_SIZE,
            "total_parameters": sum(
                p.numel() for p in model.parameters()
            ),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters()
                if p.requires_grad
            ),
        },

        "transforms": {
            "train": [
                "RandomResizedCrop(160, scale=(0.7, 1.0))",
                "RandomHorizontalFlip()",
                "ToTensor()",
                "Normalize(mean, std)",
            ],
            "validation_test": [
                "Resize((160, 160))",
                "ToTensor()",
                "Normalize(mean, std)",
            ],
            "mean": MEAN,
            "std": STD,
        },

        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE_TRAIN,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "fc_lr": 3e-3,
            "layer4_lr": (
                1e-4 if mode == "unfreeze" else None
            ),
            "loss": "CrossEntropyLoss",
            "steps_per_epoch": len(train_loader),
            "total_optimizer_steps": (
                EPOCHS * len(train_loader)
            ),
            "drop_last": False,
            "num_workers": 0,
            "shuffle": True,
            "validation_batch_size": BATCH_SIZE_EVAL,
        },

        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
            "device_name": device_info(device),
        },

        "reproducibility": {
            "split_sha256": sha256_file(SPLIT_FILE),
            **git_info(),
        },

        "results": {
            "train_loss": train_losses,
            "validation_macro_f1": validation_f1,
            "epoch_seconds": epoch_seconds,
            "training_seconds": float(training_seconds),
            "final_validation_macro_f1": float(final_f1),
            "validation_logits_shape": list(val_logits.shape),
        },
    }

    with (run_dir / "run.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nЗавершено: {run_name}")
    print(f"Фінальний validation macro-F1: {final_f1:.6f}")
    print(f"Час навчання: {training_seconds:.1f} секунд")
    print(f"Результати: {run_dir}")

    del model, optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    device = get_device()

    print("Пристрій:", device)
    print("Назва пристрою:", device_info(device))
    print("Кількість епох:", EPOCHS)
    print("Кількість запусків:", len(RUNS))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = create_dataloaders()

    print("\nРозміри вибірок:")
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))

    for mode, seed in RUNS:
        run_experiment(
            mode,
            seed,
            train_loader,
            val_loader,
            device,
        )

    print("\n" + "=" * 70)
    print("УСІ СІМ ЗАПУСКІВ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()