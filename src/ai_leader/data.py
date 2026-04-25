"""Dataset loading, validation, and label revision helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def apply_revised_labels(
    df: pd.DataFrame,
    revised_labels: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Return a copy of *df* with label updates applied by ``row_id``.

    Expected revision item format:
    ``{"row_id": int, "new_category": str?, "new_department": str?}``

    Notes:
    - Missing/invalid ``row_id`` values are ignored.
    - Only provided non-empty ``new_*`` fields are applied.
    - Accepts common typo key ``new_deparment`` as fallback.
    """
    updated = df.copy()
    if "row_id" not in updated.columns:
        raise ValueError("DataFrame must contain a 'row_id' column.")

    for item in revised_labels:
        row_id_raw = item.get("row_id")
        if row_id_raw is None:
            continue
        try:
            row_id = int(row_id_raw)
        except (TypeError, ValueError):
            continue

        row_mask = updated["row_id"] == row_id
        if not bool(row_mask.any()):
            continue

        new_category_raw = item.get("new_category")
        if new_category_raw is not None:
            new_category = str(new_category_raw).strip()
            if new_category:
                updated.loc[row_mask, "Category"] = new_category

        new_department_raw = item.get("new_department", item.get("new_deparment"))
        if new_department_raw is not None:
            new_department = str(new_department_raw).strip()
            if new_department:
                updated.loc[row_mask, "Routing to Department"] = new_department

    return updated
