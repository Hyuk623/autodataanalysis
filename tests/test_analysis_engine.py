import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app import analysis_engine  # noqa: E402


def test_infer_column_types_simple():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["a", "b", "a"],
            "value": [10.5, 20.1, 30.2],
            "date": pd.date_range("2023-01-01", periods=3),
        }
    )
    info = analysis_engine.infer_column_types(df)
    assert info["id"]["role"] in {"id", "numeric"}
    assert info["category"]["role"] == "categorical"
    assert info["value"]["role"] == "numeric"
    assert info["date"]["role"] == "datetime"


def test_run_analysis_classification():
    X, y = make_classification(n_samples=50, n_features=4, n_informative=2, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    result = analysis_engine.run_analysis(df, explicit_target="target")
    assert set(result.keys()) == {"column_info", "target_info", "problem_type", "analysis_summary"}
    assert result["problem_type"]["type"] == "classification"
    assert result["analysis_summary"]["baseline_results"] is not None
