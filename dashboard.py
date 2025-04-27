# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt # Thư viện vẽ biểu đồ tương tác tốt với Streamlit

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Dashboard Dự đoán Churn",
    page_icon="📊",
    layout="wide" # Sử dụng layout rộng hơn
)

# --- Hàm tải dữ liệu (cache để tăng tốc) ---
@st.cache_data # Cache dữ liệu để không phải đọc lại file mỗi lần tương tác
def load_data(file_path: Path) -> pd.DataFrame:
    """Tải dữ liệu dự đoán từ file CSV."""
    if not file_path.is_file():
        st.error(f"Lỗi: Không tìm thấy file dữ liệu tại '{file_path}'. Hãy chắc chắn bạn đã chạy stage 'predict'.")
        return None
    try:
        df = pd.read_csv(file_path)
        # Đảm bảo cột xác suất là số
        if 'ChurnProbability' in df.columns:
            df['ChurnProbability'] = pd.to_numeric(df['ChurnProbability'], errors='coerce')
            df.dropna(subset=['ChurnProbability'], inplace=True) # Bỏ các dòng lỗi nếu có
        else:
            st.error("Lỗi: Cột 'ChurnProbability' không tồn tại trong file dữ liệu.")
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return None

# --- Tải dữ liệu ---
DATA_FILE = Path("predictions_on_full_data.csv") # Đường dẫn tới file kết quả dự đoán
df_predictions = load_data(DATA_FILE)

# --- Tiêu đề Dashboard ---
st.title("📊 Dashboard Phân tích Dự đoán Khách hàng Rời bỏ (Churn)")
st.markdown("Dashboard này hiển thị kết quả dự đoán churn và cho phép khám phá dữ liệu.")

# --- Kiểm tra nếu dữ liệu tải thành công ---
if df_predictions is not None:

    # --- Sidebar cho bộ lọc ---
    st.sidebar.header("Bộ lọc Dữ liệu")

    # Bộ lọc theo xác suất Churn
    min_prob, max_prob = float(df_predictions['ChurnProbability'].min()), float(df_predictions['ChurnProbability'].max())
    # Sử dụng slider để chọn khoảng xác suất
    prob_range = st.sidebar.slider(
        "Chọn khoảng Xác suất Churn:",
        min_value=min_prob,
        max_value=max_prob,
        value=(min_prob, max_prob), # Giá trị mặc định là toàn bộ khoảng
        step=0.01, # Bước nhảy 0.01
        format="%.2f" # Hiển thị 2 chữ số thập phân
    )

    # Bộ lọc theo các đặc điểm khác (Ví dụ: Contract)
    # Lấy các giá trị duy nhất, bỏ qua NaN nếu có
    contract_types = ['Tất cả'] + df_predictions['Contract'].dropna().unique().tolist()
    selected_contract = st.sidebar.selectbox(
        "Chọn Loại Hợp đồng:",
        options=contract_types,
        index=0 # Mặc định chọn 'Tất cả'
    )

    # --- Lọc dữ liệu dựa trên lựa chọn ---
    df_filtered = df_predictions[
        (df_predictions['ChurnProbability'] >= prob_range[0]) &
        (df_predictions['ChurnProbability'] <= prob_range[1])
    ]
    # Áp dụng bộ lọc hợp đồng nếu không phải 'Tất cả'
    if selected_contract != 'Tất cả':
        df_filtered = df_filtered[df_filtered['Contract'] == selected_contract]

    # --- Hiển thị thông tin tổng quan ---
    st.header("Tổng quan Dữ liệu đã lọc")
    total_customers = df_filtered.shape[0]
    avg_churn_prob = df_filtered['ChurnProbability'].mean() if total_customers > 0 else 0

    col1, col2 = st.columns(2) # Tạo 2 cột
    with col1:
        st.metric("Số lượng Khách hàng", f"{total_customers:,}") # Định dạng số
    with col2:
        st.metric("Xác suất Churn Trung bình", f"{avg_churn_prob:.3f}")

    # --- Biểu đồ Phân phối Xác suất Churn ---
    st.header("Phân phối Xác suất Churn")
    if total_customers > 0:
        # Sử dụng Altair để vẽ histogram tương tác
        hist_chart = alt.Chart(df_filtered).mark_bar().encode(
            alt.X("ChurnProbability", bin=alt.Bin(maxbins=30), title="Xác suất Churn"), # Chia thành 30 khoảng
            alt.Y('count()', title="Số lượng Khách hàng"),
            tooltip=[alt.X("ChurnProbability", bin=alt.Bin(maxbins=30)), 'count()'] # Hiển thị tooltip khi di chuột
        ).properties(
            title='Biểu đồ phân phối Xác suất Churn của Khách hàng đã lọc'
        ).interactive() # Cho phép zoom/pan

        st.altair_chart(hist_chart, use_container_width=True)
    else:
        st.warning("Không có dữ liệu để hiển thị biểu đồ với bộ lọc hiện tại.")

    # --- Hiển thị Bảng Dữ liệu ---
    st.header("Chi tiết Dữ liệu Khách hàng (Đã lọc)")
    st.markdown("Bạn có thể sắp xếp bảng bằng cách nhấp vào tiêu đề cột.")

    # Chọn các cột quan trọng để hiển thị (có thể tùy chỉnh)
    columns_to_show = [
        'customerID', 'gender', 'tenure', 'Contract', 'InternetService',
        'MonthlyCharges', 'TotalCharges', 'Churn', 'ChurnProbability'
    ]
    # Chỉ giữ lại các cột tồn tại trong df_filtered
    columns_to_show = [col for col in columns_to_show if col in df_filtered.columns]

    # Hiển thị DataFrame với chiều cao cố định và thanh cuộn
    # Làm tròn cột xác suất để dễ đọc
    st.dataframe(
        df_filtered[columns_to_show].style.format({'ChurnProbability': '{:.4f}'}),
        height=400, # Giới hạn chiều cao
        use_container_width=True # Sử dụng toàn bộ chiều rộng
    )

    # --- (Tùy chọn) Thêm các phân tích khác ---
    # Ví dụ: Phân tích top N khách hàng có nguy cơ cao nhất
    st.header("Top Khách hàng có Nguy cơ Churn Cao nhất")
    num_top_customers = st.slider("Chọn số lượng khách hàng Top N:", 5, 50, 10)
    top_churners = df_filtered.nlargest(num_top_customers, 'ChurnProbability')
    st.dataframe(
        top_churners[columns_to_show].style.format({'ChurnProbability': '{:.4f}'}),
        use_container_width=True
    )

else:
    st.warning("Không thể tải dữ liệu dự đoán. Vui lòng chạy pipeline dự đoán trước.")

# --- Footer (Tùy chọn) ---
st.markdown("---")
st.caption("Dashboard tạo bằng Streamlit cho dự án Churn Prediction.")

