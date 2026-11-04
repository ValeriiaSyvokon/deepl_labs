import torch
from test import evaluate_test

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from PIL import Image

from dataset import CLASSES, TEST_DIR


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
SPLIT_FILE = ROOT / "split.csv"
FIGURES_DIR = ROOT / "figures"

CHECKPOINT_FILE = ROOT / "lab3_model.pt"

RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def load_labels():
    labels = []

    with SPLIT_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["part"] == "validation":
                labels.append(CLASSES.index(row["label"]))

    return np.array(labels)


def load_runs():
    runs = {}

    for mode, seed in RUNS:
        name = f"{mode}_seed{seed}"
        folder = RUNS_DIR / name

        with (folder / "run.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        logits = np.load(folder / "val_logits.npy")

        if logits.shape != (1498, 6):
            raise ValueError(f"Неправильна форма logits: {name}")

        runs[name] = {
            "config": config,
            "logits": logits,
        }

    return runs


def validate_runs(runs, labels):
    print("\nПЕРЕВІРКА РЕЗУЛЬТАТІВ")
    print("-" * 60)

    for name, data in runs.items():
        predictions = np.argmax(data["logits"], axis=1)

        calculated = f1_score(
            labels, predictions, average="macro"
        )

        saved = data["config"]["results"][
            "final_validation_macro_f1"
        ]

        if not np.isclose(calculated, saved, atol=1e-7):
            raise ValueError(f"Результати не збігаються: {name}")

        print(f"{name:18s} | macro-F1: {calculated:.6f}")

    print("Усі сім запусків перевірено.")


def create_plots(runs):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Train loss
    plt.figure(figsize=(10, 6))

    for name, data in runs.items():
        values = data["config"]["results"]["train_loss"]
        plt.plot(range(1, 9), values, marker="o", label=name)

    plt.xlabel("Епоха")
    plt.ylabel("Train loss")
    plt.title("Навчання моделей: Train loss")
    plt.xticks(range(1, 9))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "train_loss.png", dpi=200)
    plt.close()

    # Validation macro-F1
    plt.figure(figsize=(10, 6))

    for name, data in runs.items():
        values = data["config"]["results"]["validation_macro_f1"]
        plt.plot(range(1, 9), values, marker="o", label=name)

    plt.xlabel("Епоха")
    plt.ylabel("Validation macro-F1")
    plt.title("Навчання моделей: Validation macro-F1")
    plt.xticks(range(1, 9))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_macro_f1.png", dpi=200)
    plt.close()


def calculate_statistics(runs):
    scores = {
        name: data["config"]["results"]["final_validation_macro_f1"]
        for name, data in runs.items()
    }

    scratch = scores["scratch_seed1"]

    frozen = np.array([
        scores["frozen_seed1"],
        scores["frozen_seed2"],
        scores["frozen_seed3"],
    ])

    unfreeze = np.array([
        scores["unfreeze_seed1"],
        scores["unfreeze_seed2"],
        scores["unfreeze_seed3"],
    ])

    frozen_mean = np.mean(frozen)
    unfreeze_mean = np.mean(unfreeze)

    frozen_range = np.max(frozen) - np.min(frozen)
    unfreeze_range = np.max(unfreeze) - np.min(unfreeze)

    R = max(frozen_range, unfreeze_range)

    statistics = {
        "scratch": {
            "mean": float(scratch),
            "range": None,
        },
        "frozen": {
            "mean": float(frozen_mean),
            "range": float(frozen_range),
        },
        "unfreeze": {
            "mean": float(unfreeze_mean),
            "range": float(unfreeze_range),
        },
        "comparison": {
            "R": float(R),
            "delta_pre": float(frozen_mean - scratch),
            "delta_unf": float(unfreeze_mean - frozen_mean),
            "total_transfer_gain": float(unfreeze_mean - scratch),
        },
    }

    return statistics, scores


def print_statistics(statistics):
    print("\nПОРІВНЯННЯ РЕЖИМІВ")
    print("=" * 60)

    for mode in ("scratch", "frozen", "unfreeze"):
        item = statistics[mode]
        spread = item["range"]

        spread_text = "n/a" if spread is None else f"{spread:.6f}"

        print(
            f"{mode:10s} | "
            f"Середнє: {item['mean']:.6f} | "
            f"Розмах: {spread_text}"
        )

    comparison = statistics["comparison"]

    print("\nПРИРІСТИ")
    print("-" * 60)
    print(f"R                 = {comparison['R']:.6f}")
    print(f"Delta_pre         = {comparison['delta_pre']:+.6f}")
    print(f"Delta_unf         = {comparison['delta_unf']:+.6f}")
    print(
        f"Загальний приріст = "
        f"{comparison['total_transfer_gain']:+.6f}"
    )

    print("\nІНТЕРПРЕТАЦІЯ")

    for key, label in (
        ("delta_pre", "Попереднє навчання"),
        ("delta_unf", "Розморожування layer4"),
    ):
        value = abs(comparison[key])

        if value <= comparison["R"]:
            print(
                f"{label}: ефект не перевищує розкид запусків; "
                "його не можна впевнено відокремити від варіативності."
            )
        else:
            print(
                f"{label}: ефект перевищує розкид запусків; "
                "він вартий додаткової перевірки, "
                "але це ще не є доказом."
            )


def select_model(statistics, scores):
    # Режим із найбільшим середнім validation macro-F1
    regime = max(
        ("scratch", "frozen", "unfreeze"),
        key=lambda mode: statistics[mode]["mean"],
    )

    # Серед запусків вибраного режиму беремо найкращий
    candidates = [
        name for name in scores
        if name.startswith(regime + "_")
    ]

    selected = max(candidates, key=lambda name: scores[name])

    print("\nВИБІР МОДЕЛІ")
    print("-" * 60)
    print("Режим:", regime)
    print("Запуск:", selected)
    print(f"Validation macro-F1: {scores[selected]:.6f}")

    return {
        "regime": regime,
        "run": selected,
        "seed": int(selected.split("seed")[1]),
        "validation_macro_f1": float(scores[selected]),
    }

def validate_checkpoint(selected):
    print("\nПЕРЕВІРКА CHECKPOINT")
    print("-" * 60)

    if not CHECKPOINT_FILE.is_file():
        raise FileNotFoundError(
            f"Checkpoint не знайдено: {CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["regime"] != selected["regime"]:
        raise ValueError("Режим checkpoint не відповідає вибраному.")

    if int(checkpoint["seed"]) != selected["seed"]:
        raise ValueError("Seed checkpoint не відповідає вибраному.")

    if list(checkpoint["classes"]) != list(CLASSES):
        raise ValueError("Порядок класів у checkpoint неправильний.")

    if checkpoint["img_size"] != 160:
        raise ValueError("Неправильний розмір зображення у checkpoint.")

    if not isinstance(checkpoint["state_dict"], dict):
        raise ValueError("У checkpoint відсутній коректний state_dict.")

    print("Checkpoint:", CHECKPOINT_FILE.name)
    print("Режим:", checkpoint["regime"])
    print("Seed:", checkpoint["seed"])
    print("Метадані checkpoint перевірено.")

    return checkpoint

def evaluate_selected_model(selected):
    # Спочатку перевіряємо checkpoint
    validate_checkpoint(selected)

    # Лише після успішної перевірки відкриваємо test
    results = evaluate_test(CHECKPOINT_FILE)

    validation_f1 = selected["validation_macro_f1"]
    test_f1 = float(results["macro_f1"])

    print("\nПОРІВНЯННЯ VALIDATION І TEST")
    print("-" * 60)
    print(f"Validation macro-F1: {validation_f1:.6f}")
    print(f"Test macro-F1:       {test_f1:.6f}")
    print(f"Різниця:             {test_f1 - validation_f1:+.6f}")

    report = {
        "validation_macro_f1": float(validation_f1),
        "test_macro_f1": test_f1,
        "difference": float(test_f1 - validation_f1),
        "accuracy": float(results["accuracy"]),
        "standard_error": float(results["standard_error"]),
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    return results, report

def analyze_characteristic_errors(results):
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    probabilities = results["probabilities"]
    filenames = results["filenames"]

    # Групуємо помилки за парами справжній клас -> передбачений клас
    error_groups = {}

    for i, (true_label, predicted_label) in enumerate(
        zip(y_true, y_pred)
    ):
        if true_label != predicted_label:
            pair = (int(true_label), int(predicted_label))
            error_groups.setdefault(pair, []).append(i)

    if not error_groups:
        print("\nПомилок класифікації не знайдено.")
        return []

    # Сортування пар:
    # 1. За кількістю помилок — спаданням.
    # 2. За індексом справжнього класу.
    # 3. За індексом передбаченого класу.
    ordered_pairs = sorted(
        error_groups,
        key=lambda pair: (
            -len(error_groups[pair]),
            pair[0],
            pair[1],
        ),
    )

    # У кожній парі сортуємо файли за назвою.
    for pair in ordered_pairs:
        error_groups[pair].sort(
            key=lambda i: str(filenames[i])
        )

    # Спочатку беремо перше зображення кожної пари.
    selected_indices = []

    for pair in ordered_pairs:
        selected_indices.append(error_groups[pair][0])

        if len(selected_indices) == 5:
            break

    # Якщо пар менше п'яти, беремо наступні зображення
    # з найчастіших пар.
    depth = 1

    while len(selected_indices) < 5:
        added = False

        for pair in ordered_pairs:
            indices = error_groups[pair]

            if len(indices) > depth:
                selected_indices.append(indices[depth])
                added = True

            if len(selected_indices) == 5:
                break

        if not added:
            break

        depth += 1

    # Формуємо таблицю результатів.
    examples = []

    print("\nП'ЯТЬ ХАРАКТЕРНИХ ПОМИЛОК")
    print("=" * 90)

    print(
        f"{'Файл':35s} | "
        f"{'Справжній клас':12s} | "
        f"{'Передбачений':12s} | "
        f"{'Ймовірність':12s} | "
        f"{'Кількість':10s}"
    )

    for i in selected_indices:
        true_label = int(y_true[i])
        predicted_label = int(y_pred[i])

        pair = (true_label, predicted_label)

        example = {
            "filename": str(filenames[i]),
            "true_class": CLASSES[true_label],
            "predicted_class": CLASSES[predicted_label],
            "predicted_probability": float(
                probabilities[i, predicted_label]
            ),
            "pair_count": len(error_groups[pair]),
        }

        examples.append(example)

        print(
            f"{example['filename']:35s} | "
            f"{example['true_class']:12s} | "
            f"{example['predicted_class']:12s} | "
            f"{example['predicted_probability']:12.6f} | "
            f"{example['pair_count']:10d}"
        )

    # Зберігаємо візуалізацію.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(examples),
        figsize=(4 * len(examples), 4),
        squeeze=False,
    )

    for ax, example in zip(axes.flat, examples):
        image_path = (
            TEST_DIR
            / example["true_class"]
            / example["filename"]
        )

        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(
            f"True: {example['true_class']}\n"
            f"Pred: {example['predicted_class']}\n"
            f"P: {example['predicted_probability']:.3f}",
            fontsize=9,
        )

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "characteristic_errors.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nЗображення помилок збережено:")
    print("figures/characteristic_errors.png")

    return examples

def main():
    print("АНАЛІЗ РЕЗУЛЬТАТІВ")
    print("=" * 60)

    labels = load_labels()

    if len(labels) != 1498:
        raise ValueError("Кількість validation міток не дорівнює 1498.")

    runs = load_runs()

    validate_runs(runs, labels)
    create_plots(runs)

    statistics, scores = calculate_statistics(runs)
    print_statistics(statistics)

    selected = select_model(statistics, scores)

    test_results, test_report = evaluate_selected_model(selected)

    characteristic_errors = analyze_characteristic_errors(
        test_results
    )

    report = {
        "statistics": statistics,
        "run_scores": scores,
        "selected_model": selected,
        "test_results": test_report,
        "characteristic_errors": characteristic_errors,
    }

    with (ROOT / "analysis.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\nГрафіки збережено у figures/")
    print("Підсумки збережено в analysis.json")
    print("Результати тестування збережено в analysis.json")


if __name__ == "__main__":
    main()
