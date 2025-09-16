"""
Generic CSV loader for regression tasks.

Capabilities:
- Load arbitrary CSV files (mixed dtypes supported via structured arrays)
- Select target and feature columns by name or by index
- Returns numeric numpy arrays suitable for model consumption
"""

import csv
import numpy as np
from typing import Iterable, List, Tuple, Union, Optional, Sequence, Dict

ColumnSelector = Union[str, int]


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _to_float_array(values: np.ndarray) -> np.ndarray:
    """Convert possibly object/bytes array to float numpy array."""
    if values.dtype.kind in {"U", "S", "O"}:
        return values.astype(str).astype(float)
    return values.astype(float)


def load_data(
    filepath: str,
    *,
    target: ColumnSelector,
    features: Optional[Iterable[ColumnSelector]] = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV and return (X, y).

    - Preserves header names exactly; supports spaces and punctuation.
    - Skips rows that cannot be converted to numeric for selected columns.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header: Optional[Sequence[str]] = None
        if has_header:
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV appears empty; no header row found.")

        def resolve_indices(sel_list: Optional[Iterable[ColumnSelector]]) -> List[int]:
            if sel_list is None:
                # Use all columns except target
                if header is None:
                    raise ValueError("features=None requires header to exclude target by name")
                return [i for i, name in enumerate(header) if name != target_name]
            idxs: List[int] = []
            for sel in sel_list:
                if isinstance(sel, int):
                    idxs.append(sel)
                else:
                    if header is None:
                        raise ValueError("String feature selectors require header row present")
                    try:
                        idxs.append(header.index(sel))
                    except ValueError as exc:
                        raise ValueError(f"Feature column not found: {sel}") from exc
            return idxs

        # Resolve target
        if isinstance(target, int):
            target_idx = target
            target_name = str(target)
        else:
            if header is None:
                raise ValueError("String target selector requires header row present")
            try:
                target_idx = header.index(target)
                target_name = target
            except ValueError as exc:
                raise ValueError(f"Target column not found: {target}") from exc

        feature_idxs = resolve_indices(features)

        X_rows: List[List[float]] = []
        y_vals: List[float] = []
        # First pass: collect unique categories per feature for one-hot encoding
        categories: Dict[int, List[str]] = {}
        rows_cache: List[Sequence[str]] = []
        for row in reader:
            if not row:
                continue
            rows_cache.append(row)
            # target must be numeric; if not, skip row
            try:
                float(row[target_idx])
            except (ValueError, IndexError):
                continue
            for idx in feature_idxs:
                if idx >= len(row):
                    continue
                val = row[idx]
                # detect non-numeric
                try:
                    float(val)
                except ValueError:
                    if idx not in categories:
                        categories[idx] = []
                    if val not in categories[idx]:
                        categories[idx].append(val)

        # Second pass: build numeric feature vectors with one-hot encoding
        for row in rows_cache:
            if not row:
                continue
            try:
                y_value = float(row[target_idx])
            except (ValueError, IndexError):
                continue

            x_values: List[float] = []
            valid = True
            for idx in feature_idxs:
                if idx >= len(row):
                    valid = False
                    break
                val = row[idx]
                # If column has categories collected → treat as categorical
                if idx in categories:
                    cats = categories[idx]
                    one_hot = [1.0 if val == c else 0.0 for c in cats]
                    x_values.extend(one_hot)
                else:
                    # numeric
                    try:
                        x_values.append(float(val))
                    except ValueError:
                        valid = False
                        break

            if not valid:
                continue
            y_vals.append(y_value)
            X_rows.append(x_values)

    if not y_vals:
        raise ValueError("No valid numeric rows found for selected columns.")

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    X = _ensure_2d(X)
    return X, y

