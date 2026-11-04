from pathlib import Path
import csv
import numpy as np


# Шляхи до даних
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "seg_train" / "seg_train"
TEST_DIR = ROOT / "seg_test" / "seg_test"
OUTPUT_FILE = ROOT / "split.csv"

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

# Контрольні значення:
# (кількість зображень, сума числових частин назв файлів)
EXPECTED = {
    "buildings": {
        "train": (200, 1_996_761),
        "validation": (218, 4_837_323),
        "test": (219, 4_837_559),
    },
    "forest": {
        "train": (200, 2_032_625),
        "validation": (237, 5_276_962),
        "test": (237, 5_291_326),
    },
    "glacier": {
        "train": (200, 2_183_923),
        "validation": (276, 6_135_652),
        "test": (277, 6_157_250),
    },
    "mountain": {
        "train": (200, 1_950_345),
        "validation": (262, 5_789_535),
        "test": (263, 5_849_765),
    },
    "sea": {
        "train": (200, 1_987_613),
        "validation": (255, 5_655_068),
        "test": (255, 5_668_011),
    },
    "street": {
        "train": (200, 1_841_817),
        "validation": (250, 5_541_845),
        "test": (251, 5_547_533),
    },
}


def get_images(folder: Path) -> list[Path]:
    """Повертає JPG-файли, відсортовані за назвою."""
    if not folder.exists():
        raise FileNotFoundError(f"Не знайдено папку: {folder}")

    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() == ".jpg"),
        key=lambda p: p.name,
    )


def numeric_id(filename: str) -> int:
    """Отримує числову частину назви, наприклад 20057 з 20057.jpg."""
    return int(Path(filename).stem)


def main():
    rng_train = np.random.default_rng(0)
    rng_test = np.random.default_rng(0)

    rows = []

    for class_name in CLASSES:
        # Тренувальна частина
        train_files = get_images(TRAIN_DIR / class_name)

        if len(train_files) < 200:
            raise ValueError(
                f"У класі {class_name} недостатньо тренувальних зображень."
            )

        train_indices = np.arange(len(train_files))
        rng_train.shuffle(train_indices)

        selected_train = train_indices[:200]

        for idx in selected_train:
            rows.append({
                "file": train_files[idx].name,
                "label": class_name,
                "part": "train",
            })

        # Validation і test
        test_files = get_images(TEST_DIR / class_name)

        test_indices = np.arange(len(test_files))
        rng_test.shuffle(test_indices)

        cut = len(test_files) // 2

        validation_indices = test_indices[:cut]
        final_test_indices = test_indices[cut:]

        for idx in validation_indices:
            rows.append({
                "file": test_files[idx].name,
                "label": class_name,
                "part": "validation",
            })

        for idx in final_test_indices:
            rows.append({
                "file": test_files[idx].name,
                "label": class_name,
                "part": "test",
            })

    # Перевірка унікальності назв файлів
    filenames = [row["file"] for row in rows]

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            "Знайдено однакові назви файлів у різних частинах вибірки."
        )

    # Запис split.csv
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "label", "part"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Перевірка контрольних значень
    print("\nПеревірка поділу даних:\n")

    all_correct = True

    for class_name in CLASSES:
        print(class_name)

        for part in ["train", "validation", "test"]:
            selected = [
                row for row in rows
                if row["label"] == class_name and row["part"] == part
            ]

            count = len(selected)
            total = sum(numeric_id(row["file"]) for row in selected)

            expected_count, expected_sum = EXPECTED[class_name][part]

            correct = (
                count == expected_count
                and total == expected_sum
            )

            all_correct &= correct

            status = "OK" if correct else "ERROR"

            print(
                f"  {part:10s}: "
                f"кількість = {count:3d}, "
                f"сума = {total:>10,}  [{status}]"
            )

    print(f"\nУсього записів: {len(rows)}")
    print(f"Унікальних файлів: {len(set(filenames))}")

    if not all_correct:
        raise ValueError(
            "Контрольні значення не збігаються з методичкою!"
        )

    print("\nУсі перевірки пройдено успішно.")
    print(f"Файл збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()