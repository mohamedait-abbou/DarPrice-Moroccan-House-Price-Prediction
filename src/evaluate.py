"""Re-run error analysis for an already trained model.

This is useful when you want to inspect errors without paying for a full
retrain. It reproduces the exact train/test split used by ``src.train``.

Run with:
    python -m src.evaluate
    python -m src.evaluate --model models/darprice_model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.prep_data import OUT_FILE, ROOT
    from src.train import REPORTS_DIR, SEED, TARGET, TEST_SIZE, analyze_errors
except ImportError:  # allows `python src/evaluate.py`
    from prep_data import OUT_FILE, ROOT
    from train import REPORTS_DIR, SEED, TARGET, TEST_SIZE, analyze_errors

MODEL_PATH = ROOT / "models/darprice_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DarPrice model errors.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(OUT_FILE)
    y_log = np.log1p(df[TARGET].astype(float))
    X = df.drop(columns=[TARGET])

    _X_train, X_test, _y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=args.test_size, random_state=args.seed
    )
    y_test_mad = np.expm1(y_test_log)

    pipeline = joblib.load(args.model)
    report = analyze_errors(pipeline, X_test, y_test_mad, out_dir=args.out_dir)

    print(f"Model: {args.model.relative_to(ROOT)}")
    print(json.dumps(report["overall"], indent=2))
    print(f"\nReports written to {args.out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
