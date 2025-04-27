# api.py
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# --- SỬA IMPORT TYPING ---
from typing import List, Dict, Any, Optional, Union # Đảm bảo Any được import
# --- KẾT THÚC SỬA ---
from pathlib import Path
import sys
import joblib
import numpy as np
import shap # Import SHAP
from sklearn.utils.validation import check_is_fitted # Import để kiểm tra scaler
import traceback # Import traceback for detailed error logging

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
    title="Churn Prediction API with SHAP",
    description="API để dự đoán và giải thích khả năng rời bỏ của khách hàng.",
    version="1.1.0"
)

# --- Cấu hình CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Biến toàn cục cho artifacts và SHAP explainer ---
CONFIG_PATH = 'config/config.yaml'
MODEL = None
SCALER = None
TRAIN_COLUMNS = None
CONFIG = None
SHAP_EXPLAINER = None
SHAP_FEATURE_NAMES = None

@app.on_event("startup")
def load_model_artifacts_and_explainer():
    """Tải model, scaler, config, training columns VÀ tạo SHAP explainer."""
    global MODEL, SCALER, TRAIN_COLUMNS, CONFIG, SHAP_EXPLAINER, SHAP_FEATURE_NAMES
    try:
        print("Đang tải cấu hình và artifacts...")
        CONFIG = load_config(Path(CONFIG_PATH))
        MODEL, SCALER, TRAIN_COLUMNS = load_artifacts(CONFIG)
        print("Tải cấu hình và artifacts cơ bản thành công.")

        try:
            check_is_fitted(SCALER)
        except Exception as fit_error:
             print(f"Lỗi: Scaler chưa được fit! Lỗi: {fit_error}. Đảm bảo đã chạy stage train.")
             raise RuntimeError("Scaler is not fitted.") from fit_error

        print("Chuẩn bị dữ liệu nền cho SHAP explainer...")
        train_data_path = Path(CONFIG['data']['processed_train_path'])
        train_data = load_joblib(train_data_path)
        X_train_df = train_data['X']

        if set(X_train_df.columns) != set(TRAIN_COLUMNS):
             print("Cảnh báo: Cột trong file train đã xử lý không khớp với TRAIN_COLUMNS đã tải.")
             X_train_df = X_train_df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

        background_sample = X_train_df.sample(
            n=min(100, len(X_train_df)),
            random_state=CONFIG.get('random_state', 42)
        )

        numerical_cols = CONFIG['features'].get('numerical_cols_to_scale', [])
        cols_to_scale_in_sample = [col for col in numerical_cols if col in background_sample.columns]
        background_sample_processed = background_sample.copy()
        if cols_to_scale_in_sample:
            background_sample_processed[cols_to_scale_in_sample] = SCALER.transform(background_sample[cols_to_scale_in_sample])
            print(f"Đã chuẩn hóa {background_sample_processed.shape[0]} dòng dữ liệu nền.")
        else:
            print("Không có cột số để chuẩn hóa cho dữ liệu nền SHAP.")

        print("Kiểm tra và chuẩn hóa kiểu dữ liệu/NaN cho SHAP background data...")
        bool_cols = background_sample_processed.select_dtypes(include='bool').columns
        if not bool_cols.empty:
            print(f"Chuyển đổi các cột bool sau sang int: {bool_cols.tolist()}")
            background_sample_processed[bool_cols] = background_sample_processed[bool_cols].astype(int)

        object_cols = background_sample_processed.select_dtypes(include='object').columns
        if not object_cols.empty:
            print(f"Cảnh báo: Phát hiện các cột object sau khi xử lý: {object_cols.tolist()}.")
            cols_to_drop_for_shap = []
            for col in object_cols:
                try:
                    background_sample_processed[col] = pd.to_numeric(background_sample_processed[col], errors='raise')
                    print(f"  Cột object '{col}' đã chuyển thành số.")
                except (ValueError, TypeError):
                    print(f"  Lỗi: Không thể chuyển đổi cột object '{col}' thành số. Sẽ loại bỏ cột này cho SHAP.")
                    cols_to_drop_for_shap.append(col)
            if cols_to_drop_for_shap:
                 background_sample_processed = background_sample_processed.drop(columns=cols_to_drop_for_shap)

        background_sample_final_numeric = background_sample_processed.select_dtypes(include=np.number).copy()

        if background_sample_final_numeric.isnull().values.any():
            print("Phát hiện giá trị NaN trong dữ liệu nền SHAP. Sẽ điền bằng 0...")
            background_sample_final_numeric.fillna(0, inplace=True)
            if background_sample_final_numeric.isnull().values.any():
                 print("Lỗi: Vẫn còn NaN sau khi fillna(0).")
                 raise ValueError("Không thể xử lý NaN trong dữ liệu nền SHAP.")
            else:
                 print("Đã điền các giá trị NaN bằng 0.")
        else:
            print("Không tìm thấy giá trị NaN trong dữ liệu nền SHAP.")

        SHAP_FEATURE_NAMES = background_sample_final_numeric.columns.tolist()
        print("Kiểu dữ liệu cuối cùng của các cột cho SHAP:")
        print(background_sample_final_numeric.dtypes.to_string()) # Print all dtypes
        print(f"Số cột cuối cùng cho SHAP: {len(SHAP_FEATURE_NAMES)}")
        print(f"Đã lưu {len(SHAP_FEATURE_NAMES)} tên cột cho SHAP.")

        print("Tạo SHAP TreeExplainer...")
        SHAP_EXPLAINER = shap.TreeExplainer(MODEL, data=background_sample_final_numeric)
        print("SHAP explainer đã sẵn sàng.")

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải artifacts hoặc tạo explainer lúc khởi động: {e}")
        traceback.print_exc()
        MODEL, SCALER, TRAIN_COLUMNS, CONFIG, SHAP_EXPLAINER, SHAP_FEATURE_NAMES = None, None, None, None, None, None

# --- Pydantic Input Model ---
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
        arbitrary_types_allowed = True

# --- Pydantic Output Model (SỬA ĐỔI) ---
class PredictionResult(BaseModel):
    input_index: int
    ChurnProbability: float
    explanation: Dict[str, float] = Field(description="Đóng góp của từng feature vào dự đoán (SHAP values)")
    # --- SỬA TYPE HINT Ở ĐÂY ---
    top_features: List[Dict[str, Any]] = Field(description="Các feature ảnh hưởng nhiều nhất (tên và giá trị SHAP)")
    # --- KẾT THÚC SỬA ---

# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def read_root():
    """Endpoint gốc để kiểm tra API có hoạt động không."""
    return {"message": "Chào mừng bạn đến với API Dự đoán Churn (có giải thích SHAP)!"}

# --- Prediction Endpoint ---
@app.post("/predict", response_model=Union[PredictionResult, List[PredictionResult]], tags=["Prediction"])
async def predict_churn_and_explain(input_data: Union[CustomerData, List[CustomerData]]):
    """
    Nhận dữ liệu khách hàng, trả về xác suất churn dự đoán VÀ giải thích SHAP.
    """
    global MODEL, SCALER, TRAIN_COLUMNS, CONFIG, SHAP_EXPLAINER, SHAP_FEATURE_NAMES

    if not all([MODEL, SCALER, TRAIN_COLUMNS, CONFIG, SHAP_EXPLAINER, SHAP_FEATURE_NAMES]):
        raise HTTPException(status_code=503, detail="Lỗi: Mô hình, scaler, tên cột hoặc SHAP explainer chưa được tải.")

    try:
        # --- Prepare Input DataFrame ---
        if isinstance(input_data, list):
            input_list_of_dicts = [item.model_dump() for item in input_data]
            input_df = pd.DataFrame(input_list_of_dicts)
        else:
            input_dict = [input_data.model_dump()]
            input_df = pd.DataFrame(input_dict)
        print(f"Nhận được {input_df.shape[0]} bản ghi để dự đoán và giải thích.")

        # --- Preprocess Input Data ---
        print("Bắt đầu tiền xử lý dữ liệu đầu vào...")
        processed_df_full = preprocess_new_data(input_df, CONFIG, SCALER, TRAIN_COLUMNS)
        if processed_df_full is None:
            raise HTTPException(status_code=400, detail="Lỗi trong quá trình tiền xử lý dữ liệu đầu vào.")

        # --- Prepare Data Specifically for SHAP Explainer ---
        try:
            processed_df_for_shap = processed_df_full[SHAP_FEATURE_NAMES].copy()
            print("Finalizing data types and NaNs for SHAP input...")
            for col in processed_df_for_shap.columns:
                 if pd.api.types.is_bool_dtype(processed_df_for_shap[col]):
                      processed_df_for_shap[col] = processed_df_for_shap[col].astype(int)
                 elif not pd.api.types.is_numeric_dtype(processed_df_for_shap[col]):
                      try:
                           processed_df_for_shap[col] = pd.to_numeric(processed_df_for_shap[col], errors='coerce').fillna(0)
                           print(f"  Converted/filled non-numeric column '{col}' for SHAP.")
                      except Exception as e:
                           print(f"  Error converting column '{col}' for SHAP, filling with 0. Error: {e}")
                           processed_df_for_shap[col] = 0
            if processed_df_for_shap.isnull().values.any():
                print("Warning: NaNs detected in final SHAP input data. Filling with 0.")
                processed_df_for_shap.fillna(0, inplace=True)
        except KeyError as ke:
            print(f"Error: Missing columns required by SHAP explainer: {ke}")
            raise HTTPException(status_code=500, detail=f"Internal Error: Missing data columns for SHAP explanation.")
        except Exception as e:
             print(f"Error preparing data for SHAP: {e}")
             raise HTTPException(status_code=500, detail="Internal Error preparing data for SHAP explanation.")

        # --- Make Probability Prediction ---
        print("Thực hiện dự đoán xác suất...")
        probabilities = MODEL.predict_proba(processed_df_full)[:, 1]
        print("Dự đoán xác suất hoàn tất.")

        # --- Calculate SHAP values ---
        print("Tính toán SHAP values...")
        shap_values_output = SHAP_EXPLAINER.shap_values(processed_df_for_shap)
        if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
            shap_values_for_churn = shap_values_output[1]
        elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 2 and shap_values_output.shape[1] == len(SHAP_FEATURE_NAMES):
             shap_values_for_churn = shap_values_output
        elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 3:
             try:
                 shap_values_for_churn = shap_values_output[:,:,1]
             except IndexError:
                 print(f"Error: Cannot determine SHAP values for Churn class from 3D output shape {shap_values_output.shape}")
                 raise HTTPException(status_code=500, detail="Internal Error processing SHAP values.")
        else:
             print(f"Error: Unexpected SHAP values structure. Shape: {getattr(shap_values_output, 'shape', 'N/A')}")
             raise HTTPException(status_code=500, detail="Internal Error: Invalid SHAP values structure.")
        print("Tính toán SHAP values hoàn tất.")

        # --- Prepare Response ---
        results = []
        num_top_features = 5
        print(f"Processing {len(processed_df_for_shap)} records for response...")
        for i in range(len(processed_df_for_shap)):
            try:
                current_shap_values = shap_values_for_churn[i]
                if not isinstance(current_shap_values, np.ndarray) or current_shap_values.ndim != 1 or len(current_shap_values) != len(SHAP_FEATURE_NAMES):
                    print(f"Error: Invalid SHAP values structure for record {i}. Skipping.")
                    continue

                feature_contributions = { name: float(value) for name, value in zip(SHAP_FEATURE_NAMES, current_shap_values) }
                sorted_contributions = sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
                # Đảm bảo value trong top_features_list là float
                top_features_list = [ {"feature": name, "shap_value": round(float(value), 4)} for name, value in sorted_contributions[:num_top_features] ]

                prediction_entry = PredictionResult(
                    input_index=i,
                    ChurnProbability=round(float(probabilities[i]), 4),
                    explanation=feature_contributions,
                    top_features=top_features_list # List này giờ chứa dict[str, float]
                )
                results.append(prediction_entry)
            except Exception as loop_error:
                print(f"Error processing record {i}: {loop_error}")
                traceback.print_exc()

        if not results:
             raise HTTPException(status_code=500, detail="Failed to process any records in the request.")

        if not isinstance(input_data, list):
            return results[0]
        else:
            return results

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        print(f"Lỗi dữ liệu đầu vào hoặc xử lý: {ve}")
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Lỗi dữ liệu không hợp lệ: {ve}")
    except Exception as e:
        print(f"Lỗi không mong muốn trong quá trình dự đoán/giải thích: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {e}")


# --- Run server if script is executed directly ---
if __name__ == "__main__":
    import uvicorn
    print("Chạy FastAPI server ở chế độ debug (luôn dùng `uvicorn api:app --reload` cho dev)...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
