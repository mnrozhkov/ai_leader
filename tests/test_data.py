import pandas as pd
import pytest

from ai_leader.data import load_and_validate_dataset, validate_dataset


def test_validate_dataset_success(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "Request Text,Submission Channel,Category,Routing to Department\n"
        "Test 1,Email,Payment,Customer Support\n"
        "Test 2,Email,Order Issue,Returns\n"
        "Test 3,Email,Delivery,Logistics\n"
        "Test 4,Email,General Feedback,Product Team\n"
    )
    df = load_and_validate_dataset(str(csv_path))
    assert "row_id" in df.columns


def test_validate_dataset_missing_columns():
    df = pd.DataFrame({"Request Text": ["Test"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_invalid_category():
    df = pd.DataFrame(
        {
            "Request Text": ["Test"],
            "Submission Channel": ["Email"],
            "Category": ["Invalid"],
            "Routing to Department": ["Customer Support"],
        }
    )
    with pytest.raises(ValueError, match="Invalid category values"):
        validate_dataset(df)
