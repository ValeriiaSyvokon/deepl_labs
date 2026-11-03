"""ЛР2. Аналіз порога за збереженими прогнозами без повторного навчання.

Запуск:
    uv run python lab2_analysis.py --saved results

Очікує predictions.npz, створений lab2_starter.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions


def plot_validation_curves(
    y_true: np.ndarray, scores: np.ndarray, out_dir: Path
) -> float:
    """Будує PR- і ROC-криві лише для validation та зберігає їх."""
    out_dir.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall (повнота)")
    plt.ylabel("Precision (точність)")
    plt.title("Precision–Recall curve (validation)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Випадковий класифікатор")
    plt.xlabel("FPR (частка хибнопозитивних)")
    plt.ylabel("TPR (частка істиннопозитивних)")
    plt.title("ROC curve (validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    print(f"ROC-AUC на validation: {auc:.6f}")
    print(f"Графіки збережено в: {out_dir.resolve()}")
    return auc


def evaluate_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int | float]:
    """Обчислює TP, FP, FN, TN і C = 10*FN + FP для правила score >= threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("y_true і scores мають бути одновимірними масивами однакової довжини.")
    if len(y_true) == 0:
        raise ValueError("Масиви не можуть бути порожніми.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true має містити лише 0 і 1.")
    if not np.isfinite(scores).all():
        raise ValueError("scores мають бути скінченними числами.")

    predicted = scores >= threshold
    tp = int(np.sum((y_true == 1) & predicted))
    fp = int(np.sum((y_true == 0) & predicted))
    fn = int(np.sum((y_true == 1) & ~predicted))
    tn = int(np.sum((y_true == 0) & ~predicted))
    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "cost": 10 * fn + fp,
    }


def find_best_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    """O(n log n): сортування за оцінкою та кумулятивні суми; рівні оцінки разом."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    # Спільна перевірка форми, міток і оцінок.
    evaluate_threshold(y_true, scores, float("inf"))

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    tp_cum = np.cumsum(sorted_y == 1)
    fp_cum = np.cumsum(sorted_y == 0)
    total_positive = int(np.sum(y_true == 1))
    total_negative = int(np.sum(y_true == 0))

    rows: list[dict[str, int | float]] = [{
        "threshold": float("inf"),
        "TP": 0,
        "FP": 0,
        "FN": total_positive,
        "TN": total_negative,
        "cost": 10 * total_positive,
    }]

    # Останні індекси блоків унікальних оцінок: однакові оцінки не розділяються.
    ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    for end in ends:
        tp = int(tp_cum[end])
        fp = int(fp_cum[end])
        fn = total_positive - tp
        tn = total_negative - fp
        rows.append({
            "threshold": float(sorted_scores[end]),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "cost": 10 * fn + fp,
        })

    # Якщо C однакова, обираємо більший поріг.
    best = min(rows, key=lambda row: (row["cost"], -row["threshold"]))
    return best, rows


def brute_force_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    """Незалежна перевірка: прямий підрахунок для кожного кандидата."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    # Не використовує швидкий алгоритм: кандидати формуються окремо з унікальних оцінок.
    thresholds = [float("inf"), *np.unique(scores)[::-1].tolist()]
    rows = [evaluate_threshold(y_true, scores, float(t)) for t in thresholds]
    best = min(rows, key=lambda row: (row["cost"], -row["threshold"]))
    return best, rows


TEST_SETS = {
    "A": {
        "scores": [0.9, 0.7, 0.7, 0.4, 0.2, 0.1],
        "y": [0, 1, 0, 1, 0, 0],
    },
    "B": {
        "scores": [0.9, 0.6, 0.3],
        "y": [0, 0, 0],
    },
    "C": {
        # 0.8 має мітку 1; серед одинадцяти 0.5 перша має мітку 1,
        # інші десять — мітку 0; 0.2 має мітку 0.
        "scores": [0.8] + [0.5] * 11 + [0.2],
        "y": [1, 1] + [0] * 11,
    },
}


def check_small_test_sets(out_dir: Path) -> None:
    """Порівнює швидкий алгоритм з незалежним прямим перебором A/B/C."""
    summary_rows = []
    candidate_rows = []

    for name, dataset in TEST_SETS.items():
        y = np.asarray(dataset["y"], dtype=np.int64)
        scores = np.asarray(dataset["scores"], dtype=np.float64)
        fast_best, fast_rows = find_best_threshold(y, scores)
        brute_best, brute_rows = brute_force_threshold(y, scores)

        assert len(fast_rows) == len(brute_rows), f"{name}: різна кількість кандидатів"
        for fast, brute in zip(fast_rows, brute_rows, strict=True):
            assert fast["threshold"] == brute["threshold"], f"{name}: різні пороги"
            for key in ("TP", "FP", "FN", "TN", "cost"):
                assert fast[key] == brute[key], (
                    f"{name}: розбіжність у {key} для порога {fast['threshold']}"
                )
        assert fast_best == brute_best, f"{name}: оптимальні результати відрізняються"

        summary_rows.append({
            "Набір": name,
            "Кількість прикладів": len(y),
            "Кількість кандидатів": len(brute_rows),
            "Оптимальний поріг": brute_best["threshold"],
            "TP": brute_best["TP"],
            "FP": brute_best["FP"],
            "FN": brute_best["FN"],
            "TN": brute_best["TN"],
            "C": brute_best["cost"],
            "Перевірка": "PASS",
        })
        for row in brute_rows:
            candidate_rows.append({"Набір": name, **row, "C": row["cost"]})

        print(f"\nНабір {name}: PASS")
        print(f"Кількість кандидатів: {len(brute_rows)}")
        print(f"Оптимальний результат: {brute_best}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "small_test_set_checks.csv", index=False
    )
    pd.DataFrame(candidate_rows).drop(columns=["cost"]).to_csv(
        out_dir / "small_test_set_candidates.csv", index=False
    )
    print("\nУсі три перевірки пройдено; результати збережено в CSV.")


def evaluate_test_set(
    data: dict[str, np.ndarray], threshold: float
) -> tuple[dict[str, int | float], float | None, float | None]:
    """Оцінює test з уже вибраним на validation порогом."""
    y_test = data["test_y"]
    scores_test = data["test_scores"]
    result = evaluate_threshold(y_test, scores_test, threshold)

    precision = (
        result["TP"] / (result["TP"] + result["FP"])
        if result["TP"] + result["FP"] > 0 else None
    )
    recall = (
        result["TP"] / (result["TP"] + result["FN"])
        if result["TP"] + result["FN"] > 0 else None
    )

    print("\nОцінювання на тестовій вибірці")
    print(f"Поріг із validation: {threshold:.17g}")
    print(f"TP={result['TP']}, FP={result['FP']}")
    print(f"FN={result['FN']}, TN={result['TN']}")
    print(f"Вартість C: {result['cost']}")
    print(f"Precision: {precision:.6f}" if precision is not None else
          "Precision: не визначено (TP + FP = 0)")
    print(f"Recall: {recall:.6f}" if recall is not None else
          "Recall: не визначено (TP + FN = 0)")
    print("Матриця помилок (рядки — справжній клас, стовпці — передбачений клас):")
    print(f"[[{result['TN']}, {result['FP']}],")
    print(f" [{result['FN']}, {result['TP']}]]")
    return result, precision, recall


def show_test_error_examples(
    data: dict[str, np.ndarray], threshold: float, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Зберігає перші п'ять FP і FN за початковим індексом рядка CSV."""
    y = data["test_y"]
    scores = data["test_scores"]
    row_ids = data["test_row_ids"]
    amounts = data["test_amount"]
    if not (len(y) == len(scores) == len(row_ids) == len(amounts)):
        raise ValueError("Масиви test_y, test_scores, test_row_ids і test_amount різної довжини.")

    predicted = scores >= threshold
    fp_indices = np.flatnonzero((y == 0) & predicted)
    fn_indices = np.flatnonzero((y == 1) & ~predicted)

    def make_table(indices: np.ndarray, error_type: str) -> pd.DataFrame:
        indices = sorted(indices.tolist(), key=lambda i: int(row_ids[i]))[:5]
        return pd.DataFrame({
            "CSV row": [int(row_ids[i]) for i in indices],
            "Тип помилки": [error_type] * len(indices),
            "Мітка": [int(y[i]) for i in indices],
            "Оцінка s": [float(scores[i]) for i in indices],
            "Amount": [float(amounts[i]) for i in indices],
            "Відстань до порога": [abs(float(scores[i]) - threshold) for i in indices],
        })

    fp_table = make_table(fp_indices, "FP")
    fn_table = make_table(fn_indices, "FN")
    errors = pd.concat([fp_table, fn_table], ignore_index=True)

    print("\nПерші хибнопозитивні приклади (FP):")
    print(fp_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print("\nПерші хибнонегативні приклади (FN):")
    print(fn_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    out_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_dir / "test_error_examples.csv", index=False)
    return fp_table, fn_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Аналіз порога з уже збережених прогнозів; повторного навчання немає."
    )
    parser.add_argument(
        "--saved", type=Path, default=Path("results"),
        help="Тека зі збереженим predictions.npz (типово: results)",
    )
    args = parser.parse_args()
    out_dir = args.saved

    # спершу проходимо A, B і C.
    check_small_test_sets(out_dir)

    data = load_predictions(out_dir)
    required = {
        "validation_y", "validation_scores", "test_y", "test_scores",
        "test_row_ids", "test_amount",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise KeyError(
            f"У {out_dir / 'predictions.npz'} бракує масивів: {', '.join(missing)}. "
            "Переконайтеся, що прогнози створені актуальним lab2_starter.py."
        )

    y_validation = data["validation_y"]
    s_validation = data["validation_scores"]
    print("\nКількість validation-прикладів:", len(y_validation))
    print("Кількість шахрайських операцій:", int(np.sum(y_validation == 1)))
    print("Частка шахрайських операцій:",
          f"{np.mean(y_validation == 1):.8f}")
    print("Мінімальна оцінка:", float(np.min(s_validation)))
    print("Максимальна оцінка:", float(np.max(s_validation)))

    auc = plot_validation_curves(y_validation, s_validation, out_dir)

    best, candidates = find_best_threshold(y_validation, s_validation)
    # У CSV назва стовпця вартості — C
    candidate_df = pd.DataFrame(candidates).rename(columns={"cost": "C"})
    candidate_df.to_csv(out_dir / "threshold_candidates.csv", index=False)
    (out_dir / "threshold_star.txt").write_text(
        f"{best['threshold']:.17g}\n", encoding="utf-8"
    )
    print("Найкращий поріг t*:")
    print(best)
    print(f"Кандидатів збережено: {len(candidates)}")
    print(f"Повна таблиця: {out_dir / 'threshold_candidates.csv'}")
    print(f"Поріг збережено: {out_dir / 'threshold_star.txt'}")

    comparison = [
        {"Правило": "Стандартний поріг 0.5",
         **evaluate_threshold(y_validation, s_validation, 0.5)},
        {"Правило": "Мінімум вартості t*",
         **evaluate_threshold(y_validation, s_validation, float(best["threshold"]))},
        {"Правило": "Усі негативні +inf",
         **evaluate_threshold(y_validation, s_validation, float("inf"))},
    ]
    comparison_df = pd.DataFrame(comparison).rename(columns={"cost": "C"})
    comparison_df.to_csv(out_dir / "validation_threshold_comparison.csv", index=False)
    print("\nПорівняння порогів на validation:")
    print(comparison_df.to_string(index=False))

    # Test не використовується для вибору порога, застосовуємо t* із validation.
    test_result, precision, recall = evaluate_test_set(
        data, float(best["threshold"])
    )
    test_metrics = {
        "threshold": float(best["threshold"]),
        "TP": test_result["TP"],
        "FP": test_result["FP"],
        "FN": test_result["FN"],
        "TN": test_result["TN"],
        "C": test_result["cost"],
        "precision": precision,
        "recall": recall,
        "ROC_AUC_validation": auc,
    }
    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)
    show_test_error_examples(data, float(best["threshold"]), out_dir)

    print("\nЗбережені результати:")
    for filename in (
        "pr_curve.png", "roc_curve.png", "threshold_candidates.csv",
        "threshold_star.txt", "validation_threshold_comparison.csv",
        "small_test_set_checks.csv", "small_test_set_candidates.csv",
        "test_metrics.csv", "test_error_examples.csv",
    ):
        print(f"  {out_dir / filename}")


if __name__ == "__main__":
    main()
