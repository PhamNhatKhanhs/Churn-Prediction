# api.py
import pandas as pd
from fastapi import FastAPI, HTTPException
# --- THÊM IMPORT CORS ---
from fastapi.middleware.cors import CORSMiddleware
# --- KẾT THÚC THÊM IMPORT ---
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sys
import joblib

# --- Thêm đường dẫn src để import các module khác ---
try:
    sys.path.append(str(Path(__file__).resolve().parent / 'src'))
    from utils import load_config, load_joblib
    from models.predict_model import load_artifacts, preprocess_new_data
except ImportError as e:
    print(f"Lỗi import module: {e}. Đảm bảo bạn chạy uvicorn từ thư mục gốc dự án.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không mong muốn khi import: {e}")
    sys.exit(1)

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Churn Prediction API",
    description="API để dự đoán khả năng rời bỏ của khách hàng viễn thông.",
    version="1.0.0"
)

# --- Cấu hình CORS Middleware ---
# Cho phép tất cả các nguồn gốc (origins), phương thức (methods), và tiêu đề (headers)
# CHỈ NÊN DÙNG "*" CHO MÔI TRƯỜNG PHÁT TRIỂN/THỬ NGHIỆM
origins = [
    "*", # Cho phép tất cả
    # Trong production, bạn nên liệt kê các domain cụ thể, ví dụ:
    # "http://localhost",
    # "http://localhost:8080", # Nếu frontend chạy ở port khác
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Danh sách origins được phép
    allow_credentials=True, # Cho phép gửi cookie (nếu cần)
    allow_methods=["*"], # Cho phép tất cả các phương thức (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Cho phép tất cả các header
)
# --- Kết thúc cấu hình CORS ---


# --- Tải cấu hình và các artifact (model, scaler, columns) một lần khi khởi động ---
CONFIG_PATH = 'config/config.yaml'
MODEL = None
SCALER = None
TRAIN_COLUMNS = None
CONFIG = None

@app.on_event("startup")
def load_model_artifacts():
    """Tải model, scaler, và các cấu hình cần thiết."""
    global MODEL, SCALER, TRAIN_COLUMNS, CONFIG
    try:
        print("Đang tải cấu hình và artifacts...")
        CONFIG = load_config(Path(CONFIG_PATH))
        MODEL, SCALER, TRAIN_COLUMNS = load_artifacts(CONFIG)
        print("Tải cấu hình và artifacts thành công.")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải artifacts lúc khởi động: {e}")
        MODEL, SCALER, TRAIN_COLUMNS, CONFIG = None, None, None, None

# --- Định nghĩa cấu trúc dữ liệu đầu vào với Pydantic ---
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Any

    class Config:
        json_schema_extra = {
            "example": {
                'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
                'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
                'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
                'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 70.70, 'TotalCharges': '151.65'
            }
        }

# --- Định nghĩa Endpoint gốc (/) ---
@app.get("/", tags=["General"])
async def read_root():
    """Endpoint gốc để kiểm tra API có hoạt động không."""
    return {"message": "Chào mừng bạn đến với API Dự đoán Churn!"}

# --- Định nghĩa Endpoint dự đoán (/predict) ---
@app.post("/predict", tags=["Prediction"])
async def predict_churn(input_data: Union[CustomerData, List[CustomerData]]):
    """
    Nhận dữ liệu khách hàng và trả về xác suất churn dự đoán.
    Có thể nhận một đối tượng khách hàng hoặc một danh sách các đối tượng.
    """
    global MODEL, SCALER, TRAIN_COLUMNS, CONFIG

    if not all([MODEL, SCALER, TRAIN_COLUMNS, CONFIG]):
        raise HTTPException(status_code=503, detail="Lỗi: Mô hình hoặc cấu hình chưa được tải. Vui lòng thử lại sau.")

    try:
        if isinstance(input_data, list):
            input_list_of_dicts = [item.model_dump() for item in input_data]
            input_df = pd.DataFrame(input_list_of_dicts)
        else:
            input_dict = [input_data.model_dump()]
            input_df = pd.DataFrame(input_dict)

        print(f"Nhận được {input_df.shape[0]} bản ghi để dự đoán.")

        print("Bắt đầu tiền xử lý dữ liệu đầu vào...")
        processed_df = preprocess_new_data(input_df, CONFIG, SCALER, TRAIN_COLUMNS)

        if processed_df is None:
            raise HTTPException(status_code=400, detail="Lỗi trong quá trình tiền xử lý dữ liệu đầu vào.")

        print("Thực hiện dự đoán...")
        probabilities = MODEL.predict_proba(processed_df)[:, 1]
        print("Dự đoán hoàn tất.")

        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                "input_index": i,
                "ChurnProbability": round(prob, 4)
            })

        if not isinstance(input_data, list):
            return results[0]
        else:
            return results

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        print(f"Lỗi dữ liệu đầu vào hoặc xử lý: {ve}")
        raise HTTPException(status_code=422, detail=f"Lỗi dữ liệu không hợp lệ: {ve}")
    except Exception as e:
        print(f"Lỗi không mong muốn trong quá trình dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ trong quá trình dự đoán: {e}")

# --- Chạy server Uvicorn nếu file này được thực thi trực tiếp ---
if __name__ == "__main__":
    import uvicorn
    print("Chạy FastAPI server ở chế độ debug...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
