# tests/test_data_processing.py
import pandas as pd
import numpy as np
import pytest # Import pytest
from pathlib import Path
import sys

# --- Thêm đường dẫn src để import module cần test ---
try:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / 'src'
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))
    # Import các hàm cần test
    from data.make_dataset import handle_missing_values, encode_categorical
except ImportError as e:
    print(f"Lỗi import khi chạy test: {e}")
    handle_missing_values = None
    encode_categorical = None # Gán None nếu import lỗi


# --- Test cho handle_missing_values ---
@pytest.mark.skipif(handle_missing_values is None, reason="Không thể import hàm handle_missing_values")
def test_handle_missing_values_total_charges():
    """
    Kiểm tra hàm handle_missing_values:
    1. Chuyển đổi TotalCharges thành số.
    2. Điền giá trị NaN/rỗng trong TotalCharges bằng 0.
    """
    input_data = {
        'customerID': ['A', 'B', 'C', 'D', 'E'],
        'tenure': [0, 5, 10, 2, 1],
        'TotalCharges': [' ', '100.5', '250', '', '50.0']
    }
    input_df = pd.DataFrame(input_data)

    expected_data = {
        'customerID': ['A', 'B', 'C', 'D', 'E'],
        'tenure': [0, 5, 10, 2, 1],
        'TotalCharges': [0.0, 100.5, 250.0, 0.0, 50.0]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df['TotalCharges'] = expected_df['TotalCharges'].astype(float)

    actual_df = handle_missing_values(input_df.copy())

    pd.testing.assert_frame_equal(actual_df, expected_df)

# --- Test mới cho encode_categorical ---
@pytest.mark.skipif(encode_categorical is None, reason="Không thể import hàm encode_categorical")
def test_encode_categorical():
    """
    Kiểm tra hàm encode_categorical:
    1. Map cột target 'Churn' thành 0/1.
    2. One-Hot Encode các cột phân loại được chỉ định (hoặc tự động).
    3. Áp dụng drop_first=True cho OHE.
    """
    # 1. Chuẩn bị Input Data và Config
    input_data = {
        'gender': ['Female', 'Male', 'Female'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'Churn': ['No', 'Yes', 'No'], # Cột target
        'tenure': [1, 34, 70] # Cột số để kiểm tra không bị ảnh hưởng
    }
    input_df = pd.DataFrame(input_data)

    # Config: Để trống categorical_cols_ohe để kiểm tra auto-detection
    config = {
        'data': {'target_column': 'Churn'},
        'features': {'categorical_cols_ohe': []}
        # Nếu muốn test chỉ định cột:
        # 'features': {'categorical_cols_ohe': ['gender', 'Contract']}
    }

    # 2. Chuẩn bị Expected Output
    # Target 'Churn' map thành 0/1
    # 'gender' OHE -> gender_Male (Female là base, bị drop)
    # 'InternetService' OHE -> InternetService_Fiber optic, InternetService_No (DSL là base)
    # 'Contract' OHE -> Contract_One year, Contract_Two year (Month-to-month là base)
    expected_data = {
        'Churn': [0, 1, 0],
        'tenure': [1, 34, 70],
        'gender_Male': [False, True, False], # Female là base
        'InternetService_Fiber optic': [False, True, False], # DSL là base
        'InternetService_No': [False, False, True],
        'Contract_One year': [False, True, False], # Month-to-month là base
        'Contract_Two year': [False, False, True]
    }
    expected_df = pd.DataFrame(expected_data)
    # Chuyển đổi kiểu dữ liệu cho các cột OHE thành int (0/1) vì get_dummies trả về bool/int tùy phiên bản
    for col in expected_df.columns:
        if col not in ['Churn', 'tenure']: # Giữ nguyên kiểu của Churn và tenure
             expected_df[col] = expected_df[col].astype(int)
    # Đảm bảo thứ tự cột mong đợi (thường theo alphabet sau khi get_dummies)
    expected_df = expected_df[['Churn', 'tenure', 'Contract_One year', 'Contract_Two year', 'InternetService_Fiber optic', 'InternetService_No', 'gender_Male']]


    # 3. Gọi hàm cần test
    actual_df = encode_categorical(input_df.copy(), config)
    # Chuyển đổi kiểu dữ liệu của actual_df để khớp với expected_df
    for col in actual_df.columns:
        if col not in ['Churn', 'tenure']:
             actual_df[col] = actual_df[col].astype(int)

    # Sắp xếp cột của actual_df để đảm bảo thứ tự giống expected_df khi so sánh
    actual_df = actual_df[expected_df.columns]


    # 4. So sánh kết quả
    print("\nActual DataFrame:")
    print(actual_df)
    print("\nExpected DataFrame:")
    print(expected_df)
    print("\nActual Dtypes:")
    print(actual_df.dtypes)
    print("\nExpected Dtypes:")
    print(expected_df.dtypes)

    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=True) # Kiểm tra cả dtype

