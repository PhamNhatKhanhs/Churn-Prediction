# tests/test_data_processing.py
import pandas as pd
import numpy as np
import pytest # Import pytest
from pathlib import Path
import sys

# --- Thêm đường dẫn src để import module cần test ---
# Cách này giúp chạy test từ thư mục gốc dự án bằng lệnh pytest
try:
    # Thêm thư mục gốc của dự án vào sys.path
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / 'src'
    sys.path.insert(0, str(project_root)) # Thêm gốc trước để ưu tiên import từ src
    sys.path.insert(0, str(src_path))
    # Import hàm cần test
    from data.make_dataset import handle_missing_values
except ImportError as e:
    print(f"Lỗi import khi chạy test: {e}")
    print("Hãy đảm bảo bạn chạy pytest từ thư mục gốc của dự án.")
    # Hoặc bạn có thể cần cài đặt dự án ở chế độ editable: pip install -e .
    # Nếu vẫn lỗi, kiểm tra lại cấu trúc thư mục và sys.path
    handle_missing_values = None # Gán None để tránh lỗi nếu import thất bại


# --- Viết hàm test ---
# Tên hàm test cũng bắt đầu bằng test_
# Sử dụng @pytest.mark.skipif để bỏ qua test nếu import hàm thất bại
@pytest.mark.skipif(handle_missing_values is None, reason="Không thể import hàm handle_missing_values")
def test_handle_missing_values_total_charges():
    """
    Kiểm tra hàm handle_missing_values:
    1. Chuyển đổi TotalCharges thành số.
    2. Điền giá trị NaN/rỗng trong TotalCharges bằng 0.
    """
    # 1. Chuẩn bị dữ liệu đầu vào (Input Data)
    input_data = {
        'customerID': ['A', 'B', 'C', 'D', 'E'],
        'tenure': [0, 5, 10, 2, 1],
        'TotalCharges': [' ', '100.5', '250', '', '50.0'] # Có giá trị rỗng và chuỗi rỗng
        # Thêm các cột khác nếu hàm handle_missing_values có xử lý chúng
    }
    input_df = pd.DataFrame(input_data)

    # 2. Chuẩn bị dữ liệu đầu ra mong đợi (Expected Output)
    expected_data = {
        'customerID': ['A', 'B', 'C', 'D', 'E'],
        'tenure': [0, 5, 10, 2, 1],
        'TotalCharges': [0.0, 100.5, 250.0, 0.0, 50.0] # Giá trị rỗng/trống đã được điền 0.0 và chuyển thành float
    }
    expected_df = pd.DataFrame(expected_data)
    # Đảm bảo kiểu dữ liệu cột TotalCharges là float
    expected_df['TotalCharges'] = expected_df['TotalCharges'].astype(float)

    # 3. Gọi hàm cần test
    # Tạo bản sao để tránh thay đổi input_df gốc nếu hàm có inplace=True (mặc dù hàm đã sửa không dùng inplace)
    actual_df = handle_missing_values(input_df.copy())

    # 4. So sánh kết quả thực tế với kết quả mong đợi
    # Sử dụng hàm tiện ích của pandas để so sánh DataFrame, nó xử lý kiểu dữ liệu và NaN tốt hơn assert df1 == df2
    pd.testing.assert_frame_equal(actual_df, expected_df)

# --- Thêm các hàm test khác cho các trường hợp khác của hàm này hoặc các hàm khác ---
# Ví dụ: test_encode_categorical, test_preprocess_data...
# def test_encode_categorical_mapping():
#     # ... chuẩn bị input/output cho việc map target ...
#     pass

# def test_encode_categorical_ohe():
#     # ... chuẩn bị input/output cho one-hot encoding ...
#     pass

