import pandas as pd
from scalpers_time_machine.data_loader import load_data

def test_load_data_returns_dataframe():
    df = load_data()
    df = df.sort_index()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "DataFrame should not be empty"
    assert "company_prefix" in df.columns, "'company_prefix' should be a column"
    assert df.index.is_monotonic_increasing, "Index should be sorted"