"""
Unit tests for data loading, schema validation, and dtype casting.
"""

import gzip
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.load import (
    RAW_COLUMNS,
    FLOAT_COLUMNS,
    OBJECT_COLUMNS,
    SCHEMA_DTYPES,
    SchemaValidationError,
    validate_schema,
    drop_metadata_rows,
    cast_dtypes,
    load_chunks,
    load_raw,
    resolve_data_path,
)


def create_synthetic_sample(n_rows: int = 5) -> pd.DataFrame:
    """Create a minimal synthetic DataFrame with all 151 columns."""
    data = {}
    for col in RAW_COLUMNS:
        if col in FLOAT_COLUMNS:
            data[col] = [float(i + 1) for i in range(n_rows)]
        else:
            if col == "id":
                data[col] = [f"{1000000 + i}" for i in range(n_rows)]
            elif col == "loan_status":
                statuses = ["Fully Paid", "Charged Off", "Current", "Default", "In Grace Period"]
                data[col] = [statuses[i % len(statuses)] for i in range(n_rows)]
            else:
                data[col] = [f"val_{i}" for i in range(n_rows)]
    return pd.DataFrame(data)


def test_schema_validation_success():
    """Test that schema validation succeeds when all 151 expected columns are present."""
    df = create_synthetic_sample(5)
    # Should complete without error
    validate_schema(df)


def test_schema_validation_missing_column_fails_loudly():
    """Test that schema validation fails loudly if an expected column is missing."""
    df = create_synthetic_sample(5)
    df_missing = df.drop(columns=["loan_status", "loan_amnt"])

    with pytest.raises(SchemaValidationError) as excinfo:
        validate_schema(df_missing)

    err_msg = str(excinfo.value)
    assert "SCHEMA VALIDATION FAILED" in err_msg
    assert "'loan_status' missing" in err_msg
    assert "'loan_amnt' missing" in err_msg


def test_schema_validation_renamed_column_detection():
    """Test that renamed columns are detected and reported with helpful suggestions."""
    df = create_synthetic_sample(5)
    df_renamed = df.rename(columns={"loan_amnt": "loan_amount"})

    with pytest.raises(SchemaValidationError) as excinfo:
        validate_schema(df_renamed)

    err_msg = str(excinfo.value)
    assert "'loan_amnt' missing" in err_msg
    assert "did you rename it to 'loan_amount'?" in err_msg


def test_strict_schema_validation_extra_column():
    """Test that strict mode catches unexpected extra columns."""
    df = create_synthetic_sample(5)
    df["extra_surprise_col"] = 123

    # Normal mode allows extra columns
    validate_schema(df, strict=False)

    # Strict mode fails loudly
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_schema(df, strict=True)

    assert "STRICT SCHEMA VALIDATION FAILED" in str(excinfo.value)
    assert "extra_surprise_col" in str(excinfo.value)


def test_drop_metadata_rows():
    """Test that footer/metadata rows are dropped and legitimate loans retained."""
    df = create_synthetic_sample(4)

    # Add 2 metadata rows similar to the actual Lending Club dataset footer rows
    meta_row1 = {col: np.nan for col in RAW_COLUMNS}
    meta_row1["id"] = "Total amount funded in policy code 1: 1465324575"
    meta_row1["loan_status"] = np.nan

    meta_row2 = {col: np.nan for col in RAW_COLUMNS}
    meta_row2["id"] = "Loans that do not meet the credit policy"
    meta_row2["loan_status"] = np.nan

    df_with_meta = pd.concat([df, pd.DataFrame([meta_row1, meta_row2])], ignore_index=True)
    assert len(df_with_meta) == 6

    cleaned = drop_metadata_rows(df_with_meta)
    assert len(cleaned) == 4
    assert not cleaned["id"].str.contains("Total amount funded").any()
    assert cleaned["loan_status"].isna().sum() == 0


def test_cast_dtypes():
    """Test that numerical columns are cast to float64 and object to object."""
    df = create_synthetic_sample(3)

    # Corrupt some numeric dtypes to strings
    df["loan_amnt"] = ["1000.5", "2500.0", "corrupted_text"]
    df["annual_inc"] = [50000, 75000, 100000]  # integers

    casted = cast_dtypes(df)

    assert casted["loan_amnt"].dtype == "float64"
    assert casted["loan_amnt"].iloc[0] == 1000.5
    assert casted["loan_amnt"].iloc[1] == 2500.0
    assert np.isnan(casted["loan_amnt"].iloc[2])  # coerced to NaN

    assert casted["annual_inc"].dtype == "float64"
    assert casted["term"].dtype == "object"


def test_chunked_loader_with_synthetic_gz(tmp_path: Path):
    """Test that load_chunks correctly streams and processes a gzipped CSV file."""
    df = create_synthetic_sample(10)
    gz_file = tmp_path / "synthetic_accepted.csv.gz"

    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)

    # Stream chunks of size 3
    chunks = list(load_chunks(filepath=gz_file, chunksize=3, validate=True, cast=True))

    assert len(chunks) == 4  # 3 + 3 + 3 + 1 = 10 rows
    total_rows = sum(len(c) for c in chunks)
    assert total_rows == 10

    # Ensure all chunks are properly cast and typed
    for chunk in chunks:
        assert chunk.shape[1] == 151
        assert chunk["loan_amnt"].dtype == "float64"
        assert chunk["term"].dtype == "object"


def test_resolve_data_path_nonexistent():
    """Test that resolve_data_path raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        resolve_data_path("/non/existent/path/data.csv")
