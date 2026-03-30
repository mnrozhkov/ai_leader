"""Dataset loading and validation."""

from __future__ import annotations

import pandas as pd

from .config import CATEGORY_VALUES, CONFIDENCE_LEVELS, DEPARTMENT_VALUES

REQUIRED_COLUMNS = [
    "Request Text",
    "Submission Channel",
    "Category",
    "Routing to Department",
]

OPTIONAL_COLUMNS = [
    "Order History",
    "Order ID",
    "Related to order",
    "Timestamp",
    "[Human] Initial Response",
    "Risks (yes/no)",
]


def load_dataset(path_or_url: str) -> pd.DataFrame:
    return pd.read_csv(path_or_url)


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    invalid_categories = sorted(
        {value for value in df["Category"].dropna().unique() if value not in CATEGORY_VALUES}
    )
    if invalid_categories:
        raise ValueError(f"Invalid category values: {invalid_categories}")
    missing_categories = [
        value for value in CATEGORY_VALUES if value not in df["Category"].unique()
    ]
    if missing_categories:
        raise ValueError(f"Missing category values: {missing_categories}")

    invalid_departments = sorted(
        {
            value
            for value in df["Routing to Department"].dropna().unique()
            if value not in DEPARTMENT_VALUES
        }
    )
    if invalid_departments:
        raise ValueError(f"Invalid department values: {invalid_departments}")
    missing_departments = [
        value for value in DEPARTMENT_VALUES if value not in df["Routing to Department"].unique()
    ]
    if missing_departments:
        raise ValueError(f"Missing department values: {missing_departments}")

    if "Confidence" in df.columns:
        invalid_confidence = sorted(
            {
                value
                for value in df["Confidence"].dropna().unique()
                if value not in CONFIDENCE_LEVELS
            }
        )
        if invalid_confidence:
            raise ValueError(f"Invalid confidence values: {invalid_confidence}")


def load_and_validate_dataset(path_or_url: str) -> pd.DataFrame:
    df = load_dataset(path_or_url)
    validate_dataset(df)
    if "row_id" not in df.columns:
        df = df.reset_index(drop=True).copy()
        df["row_id"] = df.index
    return df
