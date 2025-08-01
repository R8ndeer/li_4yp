import pandas as pd
from typing import Any
from pathlib import Path


data_path = Path().resolve().parent / "data"
if not data_path.exists():
    raise FileNotFoundError(f"The data directory {data_path} does not exist.")


def load_data(file_name: str) -> pd.DataFrame:
    """Load YUV dataset (CSV)."""
    file_path = data_path / file_name
    if file_path.exists():
        print(f"Loaded data from {file_path}")
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_name} does not exist in the data directory.")
    

def split_single_shade_code(code: Any) -> pd.Series:
    """Split shade into Base, Primary, Secondary, and Tertiary."""
    parts = str(code).split('.', maxsplit=1)
    if len(parts) > 2:
        raise ValueError(f"Invalid shade code format: {code}")
    
    try:
        base = int(parts[0]) if parts[0] else 0
    except Exception:
        #! print(f"Non-integer base value in shade code, skipping: {code}")
        return pd.Series(
        [0, 0, 0, 0],
        index=['Base', 'Primary', 'Secondary', 'Tertiary'],
        dtype=object
        )

    if len(parts) > 1:
        additional = list(parts[1])
        primary = int(additional[0]) if len(additional) > 0 else 0
        secondary = int(additional[1]) if len(additional) > 1 else 0
        tertiary = int(additional[2]) if len(additional) > 2 else 0
        if len(additional) > 3:
            raise ValueError(f"Too many components in shade code: {code}")
    else:
        primary, secondary, tertiary = 0, 0, 0
    return pd.Series(
        [base, primary, secondary, tertiary],
        index=['Base', 'Primary', 'Secondary', 'Tertiary'],
        dtype=object
        )


def split_shade_codes(df: pd.DataFrame, column_name: str = "Shade") -> None:
    """Apply the split_single_shade_code() function to all columns of a DataFrame."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    df.loc[:, ['Base', 'Primary', 'Secondary', 'Tertiary']] = df[column_name].apply(split_single_shade_code)
    df.replace(0, None, inplace=True)
    df.drop(columns=[column_name], inplace=True)


def split_lab(df: pd.DataFrame, column_name: str = "Corrected LAB") -> None:
    """Split the Lab values into L, a, b components."""
    if column_name not in df.columns:
        raise ValueError(f"[split_lab] Column '{column_name}' does not exist in the DataFrame.")
    
    df.loc[:, ['L', 'a', 'b']] = df[column_name].apply(lambda x: pd.Series(str(x).split(',')))
    df.drop(columns=[column_name], inplace=True)
    df.drop(columns=['Dominant LAB'], inplace=True, errors='ignore')


def preprocess_lab_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by splitting shade codes and Lab values.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess, usually loaded from a CSV file using load_data().
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame with shade codes and Lab values split into separate columns.
    """
    split_shade_codes(df)
    split_lab(df)
    return df


def preprocess_formula_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by splitting shade codes and Lab values.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess, usually loaded from a CSV file using load_data().
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame with formula and shade.
    """
    columns = [f'AA0{i}' for i in range(1, 8)] + ['shade']
    df_new = df.loc[:, columns]
    split_shade_codes(df_new, column_name='shade')
    return df_new