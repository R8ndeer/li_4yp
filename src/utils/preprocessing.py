import pandas as pd
from typing import Any, Tuple
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
    

def parse_shade(code: Any) -> Tuple[int, int, int, int]:
    """Split shade into Base, Primary, Secondary, and Tertiary."""
    parts = str(code).split('.', maxsplit=1)
    if len(parts) > 2:
        raise ValueError(f"Invalid shade code format: {code}")
    
    # Skip non-integer base value (e.g. "10 (1/4)")
    try:
        base = int(parts[0]) if parts[0] else 0
    except:
        #! print(f"Non-integer base value in shade code, skipping: {code}")
        return (0, 0, 0, 0)

    if len(parts) > 1:
        additional = list(parts[1])
        primary = int(additional[0]) if len(additional) > 0 else 0
        secondary = int(additional[1]) if len(additional) > 1 else 0
        tertiary = int(additional[2]) if len(additional) > 2 else 0
        if len(additional) > 3:
            raise ValueError(f"[parse_shade] Too many components in shade code: {code}")
    else:
        primary, secondary, tertiary = 0, 0, 0
    return (base, primary, secondary, tertiary)


def split_shade(df: pd.DataFrame, column_name: str = "Shade") -> None:
    """Apply the parse_shade() function to all columns of a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess, usually loaded from a CSV file using load_data().
        column_name (str): The name of the column containing shade codes. Defaults to "Shade".
    """
    if column_name not in df.columns:
        raise ValueError(f"[split_shade] Column '{column_name}' does not exist in the DataFrame.")
    
    base, p1, p2, p3 = zip(*df[column_name].apply(parse_shade))  # unpack series of tuples
    df['Base'] = base
    df['Primary'] = p1
    df['Secondary'] = p2
    df['Tertiary'] = p3
    df.drop(columns=[column_name], inplace=True)


def split_lab(df: pd.DataFrame, column_name: str = "Corrected LAB") -> None:
    """Split the Lab values into L, a, b components.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess, usually loaded from a CSV file using load_data().
        column_name (str): The name of the column containing Lab values. Defaults to "Corrected LAB".
    """
    if column_name not in df.columns:
        raise ValueError(f"[split_lab] Column '{column_name}' does not exist in the DataFrame.")
    
    df.loc[:, ['L', 'a', 'b']] = df[column_name].apply(lambda x: pd.Series(str(x).split(','))).values
    df.drop(columns=[column_name], inplace=True)
    df.drop(columns=['Dominant LAB'], inplace=True, errors='ignore')


def preprocess_lab_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by splitting shade codes and Lab values.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess, usually loaded from a CSV file using load_data().
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame with shade codes and Lab values split into separate columns.
    """
    split_shade(df)
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
    df = df.loc[:, columns]
    split_shade(df, column_name='shade')
    return df


def merge_on_shade(lab_df: pd.DataFrame, formula_df: pd.DataFrame) -> pd.DataFrame:
    """Merge two DataFrames on shade columns.
    
    Args:
        lab_df (pd.DataFrame): The DataFrame containing Lab data.
        formula_df (pd.DataFrame): The DataFrame containing formula data.

    Returns:
        pd.DataFrame: The merged DataFrame containing both Lab and formula data.
    """
    return pd.merge(
        lab_df,
        formula_df,
        on=['Base', 'Primary', 'Secondary', 'Tertiary'],
        how='inner',
        suffixes=['_lab', '_formula']
    )