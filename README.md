# Dự án Dự đoán Churn Khách hàng Viễn thông (End-to-End Machine Learning)

**Phiên bản:** 1.1.0 (Tích hợp SHAP & CI/CD)

## Giới thiệu

Dự án này trình bày một giải pháp Machine Learning đầu cuối (End-to-End) nhằm giải quyết bài toán **dự đoán khách hàng rời bỏ (Churn Prediction)** cho một công ty viễn thông giả định. Việc mất khách hàng (churn) gây tổn thất doanh thu đáng kể và chi phí thu hút khách hàng mới thường cao hơn chi phí giữ chân khách hàng hiện tại. Do đó, khả năng xác định sớm các khách hàng có nguy cơ churn cao là cực kỳ quan trọng để triển khai các chiến lược giữ chân (retention) hiệu quả và tối ưu hóa nguồn lực.

Dự án này không chỉ dừng lại ở việc xây dựng mô hình dự đoán mà còn bao gồm các khía cạnh quan trọng của một quy trình ML hoàn chỉnh:

* Phân tích dữ liệu sâu sắc để hiểu các yếu tố lịch sử.
* Xây dựng pipeline huấn luyện tự động và có thể tái lập.
* Tối ưu hóa hiệu năng mô hình thông qua tinh chỉnh siêu tham số.
* Theo dõi và quản lý các thử nghiệm bằng MLflow.
* Giải thích các dự đoán của mô hình bằng kỹ thuật XAI (SHAP).
* Đóng gói mô hình thành một API backend (FastAPI) có thể truy cập qua web.
* Xây dựng giao diện frontend tương tác để sử dụng API.
* Thiết lập quy trình CI/CD để tự động hóa kiểm thử, build và triển khai.
* Triển khai ứng dụng lên môi trường cloud (Render).

**Dữ liệu gốc:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) từ Kaggle.

## Các Tính năng Kỹ thuật Nổi bật

1.  **Pipeline Dữ liệu & Huấn luyện (`main.py`, `src/`)**:
    * Tự động hóa các bước: Tải dữ liệu thô, tiền xử lý (xử lý missing values, encoding, scaling), chia tập train/test.
    * Tích hợp huấn luyện mô hình (hiện tại là RandomForestClassifier) với **tinh chỉnh siêu tham số tự động** bằng `GridSearchCV`, tập trung tối ưu chỉ số **Recall** cho lớp Churn.
    * Lưu trữ các artifact quan trọng (mô hình đã huấn luyện, scaler) bằng `joblib`.

2.  **Theo dõi Thử nghiệm (`MLflow`)**:
    * Mỗi lần chạy pipeline huấn luyện (`main.py train` hoặc `full`) sẽ tự động tạo một "run" trong MLflow.
    * Ghi lại chi tiết:
        * Tham số cấu hình (tên model, kích thước test set...).
        * Siêu tham số tốt nhất tìm được bởi GridSearchCV.
        * Điểm Cross-Validation tốt nhất (theo Recall).
        * Các chỉ số đánh giá cuối cùng trên tập test (Accuracy, Precision, Recall, F1, AUC).
        * Các artifact: Mô hình đã huấn luyện (`model`), scaler, file báo cáo phân loại, biểu đồ confusion matrix, biểu đồ feature importance.
    * Cho phép dễ dàng so sánh các thử nghiệm và quản lý vòng đời mô hình thông qua MLflow UI.

3.  **Giải thích Dự đoán (XAI với `SHAP`)**:
    * Sử dụng `shap.TreeExplainer` để tính toán SHAP values cho mô hình RandomForest.
    * Cung cấp khả năng giải thích cho từng dự đoán cá nhân: xác định các đặc trưng (features) đóng góp nhiều nhất vào việc tăng hoặc giảm xác suất churn.
    * Tích hợp vào API để trả về thông tin giải thích cùng với xác suất dự đoán.

4.  **API Backend (`api.py` với `FastAPI`)**:
    * Cung cấp endpoint `/predict` (POST) để nhận dữ liệu của một hoặc nhiều khách hàng (dạng JSON).
    * Tự động tải mô hình, scaler, và SHAP explainer khi khởi động.
    * Thực hiện tiền xử lý dữ liệu đầu vào nhất quán với quá trình huấn luyện.
    * Trả về kết quả dự đoán bao gồm:
        * `ChurnProbability`: Xác suất churn dự đoán.
        * `explanation`: Dictionary chứa SHAP value cho tất cả các feature.
        * `top_features`: Danh sách N feature có ảnh hưởng lớn nhất (cả dương và âm).
    * Tích hợp CORS Middleware để cho phép frontend tương tác.
    * Cung cấp tài liệu API tự động (Swagger UI tại `/docs` và ReDoc tại `/redoc`).

5.  **Giao diện Frontend (`frontend.html`)**:
    * Xây dựng bằng HTML, Tailwind CSS (cho styling hiện đại & responsive), và JavaScript.
    * Cho phép người dùng nhập thông tin chi tiết của một khách hàng và nhận dự đoán + giải thích SHAP trực quan (dạng biểu đồ cột ngang Chart.js).
    * Cung cấp chức năng **Phân tích "What-If"**: Cho phép thay đổi giá trị `Contract` và `tenure` và xem xác suất churn dự đoán thay đổi như thế nào ngay lập tức.
    * Cho phép **Upload file CSV** chứa nhiều khách hàng, gọi API theo lô và hiển thị kết quả dự đoán trong bảng.

6.  **Dockerization (`Dockerfile`, `.dockerignore`)**:
    * Đóng gói ứng dụng API FastAPI thành một Docker image độc lập, chứa đủ môi trường và thư viện cần thiết.
    * Đảm bảo tính nhất quán và dễ dàng triển khai trên các môi trường khác nhau.

7.  **CI/CD & Deployment (`GitHub Actions`, `GHCR`, `Render`)**:
    * Workflow GitHub Actions (`.github/workflows/ci.yml`) tự động kích hoạt khi push lên nhánh `master`.
    * **CI (Continuous Integration):**
        * Kiểm tra code style bằng `flake8`.
        * Chạy unit test bằng `pytest` (hiện tại chưa có nhiều test).
        * Build Docker image để xác nhận Dockerfile hoạt động.
    * **CD (Continuous Deployment):**
        * Đăng nhập và đẩy Docker image đã build lên GitHub Container Registry (GHCR) với tag `latest` và tag theo commit SHA.
        * Tự động triển khai phiên bản image mới nhất lên **Render** thông qua cơ chế Auto-Deploy của Render (theo dõi tag `latest` trên GHCR).

## Công nghệ Sử dụng

* **Ngôn ngữ:** Python 3.11+
* **Thư viện Python chính:**
    * `pandas`: Xử lý và phân tích dữ liệu.
    * `numpy`: Tính toán số học.
    * `scikit-learn`: Xây dựng mô hình ML (RandomForest), tiền xử lý (StandardScaler), đánh giá (metrics), tuning (GridSearchCV).
    * `mlflow`: Theo dõi thử nghiệm ML.
    * `shap`: Giải thích mô hình ML.
    * `fastapi`: Xây dựng API backend.
    * `uvicorn`: ASGI server để chạy FastAPI.
    * `pydantic`: Xác thực dữ liệu cho API.
    * `joblib`: Lưu và tải mô hình/scaler.
    * `PyYAML`: Đọc file cấu hình YAML.
    * `pytest`: Framework cho unit testing.
    * `flake8`: Kiểm tra code style (linting).
    * `requests`: (Trong ví dụ client) Gửi yêu cầu HTTP đến API.
    * `streamlit`: (Trong dashboard tùy chọn) Xây dựng dashboard tương tác.
    * `imbalanced-learn`: (Nếu dùng SMOTE) Xử lý mất cân bằng dữ liệu.
* **Frontend:**
    * HTML5
    * Tailwind CSS (v3.x qua CDN)
    * JavaScript (ES6+)
    * Chart.js (v4.x qua CDN)
    * Font Awesome (v6.x qua CDN - cho icons)
* **MLOps & Hạ tầng:**
    * `venv`: Quản lý môi trường ảo Python.
    * `Git` & `GitHub`: Quản lý phiên bản và lưu trữ code.
    * `Docker` & `Docker Compose` (cho Airflow nếu dùng): Containerization.
    * `GitHub Actions`: CI/CD.
    * `GitHub Container Registry (GHCR)`: Lưu trữ Docker image.
    * `Render`: Nền tảng Cloud để triển khai API (PaaS).
    * `(Tùy chọn)` `Task Scheduler (Windows)` / `Cron (Linux)`: Lập lịch chạy đơn giản.

## Cấu trúc Thư mục

churn_prediction_project/├── .github/workflows/      # Chứa file workflow của GitHub Actions (ci.yml)├── .gitignore              # Các file/thư mục Git bỏ qua├── .dockerignore           # Các file/thư mục Docker bỏ qua khi build image├── Dockerfile              # Định nghĩa cách build Docker image cho API├── README.md               # File này├── api.py                  # Code ứng dụng FastAPI backend├── config/                 # Chứa file cấu hình│   └── config.yaml├── data/│   ├── raw/                # Dữ liệu thô ban đầu (WA_Fn-UseC_-Telco-Customer-Churn.csv)│   └── processed/          # Dữ liệu đã xử lý (train.joblib, test.joblib)├── frontend.html           # Giao diện người dùng web tĩnh├── main.py                 # Script chính để chạy các stage của pipeline├── models/                 # Lưu các artifact đã huấn luyện (model.joblib, scaler.joblib)├── notebooks/              # (Tùy chọn) Chứa các Jupyter Notebook cho EDA, thử nghiệm├── reports/                # Lưu các kết quả đánh giá, hình ảnh│   ├── figures/            # Biểu đồ (confusion matrix, feature importance...)│   └── metrics/            # File metrics (metrics.json, classification_report.txt)├── requirements.txt        # Danh sách các thư viện Python cần thiết├── src/                    # Mã nguồn Python được module hóa│   ├── init.py│   ├── data/               # Scripts xử lý dữ liệu (make_dataset.py)│   ├── models/             # Scripts huấn luyện, đánh giá, dự đoán│   └── utils.py            # Các hàm tiện ích├── tests/                  # Chứa các unit test (test_data_processing.py, ...)└── venv/                   # Thư mục môi trường ảo Python (trong .gitignore)
## Cài đặt và Thiết lập

1.  **Clone Repository:**
    ```bash
    git clone <URL_repository_cua_ban>
    cd churn_prediction_project
    ```
2.  **Cài đặt Python:** Đảm bảo bạn đã cài đặt Python (khuyến nghị 3.11 hoặc phiên bản tương thích được chỉ định trong `Dockerfile` và `ci.yml`).
3.  **Tạo và Kích hoạt Môi trường ảo:**
    ```bash
    # Tạo môi trường ảo
    python -m venv venv
    # Kích hoạt (Windows PowerShell)
    .\venv\Scripts\Activate.ps1
    # Kích hoạt (Linux/macOS/Git Bash)
    # source venv/bin/activate
    ```
4.  **Cài đặt Thư viện:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
5.  **Tải Dữ liệu Gốc:** Tải file `WA_Fn-UseC_-Telco-Customer-Churn.csv` từ Kaggle (hoặc nguồn khác) và đặt vào thư mục `data/raw/`.
6.  **(Tùy chọn) Cài đặt Docker Desktop:** Nếu bạn muốn build và chạy Docker image cục bộ hoặc cần thiết cho các bước MLOps nâng cao.

## Hướng dẫn Sử dụng

**1. Huấn luyện Mô hình và Theo dõi:**

* Chạy toàn bộ pipeline (preprocess, train, evaluate, log MLflow):
    ```bash
    python main.py full
    ```
* Kết quả huấn luyện, model, scaler sẽ được lưu vào các thư mục `data/processed/`, `models/`, `reports/`.
* Mở MLflow UI để xem chi tiết thử nghiệm:
    ```bash
    mlflow ui
    ```
    Truy cập `http://localhost:5000`.

**2. Chạy API Backend (Cục bộ):**

* Đảm bảo đã chạy `python main.py full` ít nhất một lần để tạo artifacts.
* Khởi chạy server FastAPI:
    ```bash
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
    ```
* Truy cập API docs: `http://127.0.0.1:8000/docs`

**3. Sử dụng Giao diện Frontend (Cục bộ):**

* Đảm bảo API Backend đang chạy (Bước 2).
* Mở file `frontend.html` bằng trình duyệt web.
* Sử dụng form để dự đoán đơn lẻ, xem giải thích SHAP, thử "What-If".
* Sử dụng chức năng upload để dự đoán cho file CSV.

**4. Chạy Dự đoán theo Lô (Command Line):**

* Chuẩn bị file `input.csv` với dữ liệu cần dự đoán.
* Chạy lệnh:
    ```bash
    python main.py predict --input-file input.csv --output-file predictions.csv
    ```

**5. Sử dụng API đã Triển khai (Render):**

* Lấy URL công khai của dịch vụ từ dashboard Render (ví dụ: `https://your-service-name.onrender.com`).
* Truy cập `URL/docs` để xem Swagger UI.
* Sử dụng các công cụ như `curl`, Postman, hoặc thư viện `requests` trong Python để gửi yêu cầu POST đến `URL/predict` với dữ liệu khách hàng.
* Cập nhật biến `apiUrl` trong `frontend.html` thành URL Render để frontend cục bộ gọi API trên cloud.

## Quy trình Làm việc (Workflow)

1.  **Thu thập & Khám phá:** Dữ liệu thô được tải và phân tích (EDA) để hiểu đặc điểm và các yếu tố ban đầu.
2.  **Tiền xử lý:** Dữ liệu được làm sạch, xử lý giá trị thiếu, mã hóa biến phân loại (OHE), và chuẩn hóa biến số. Dữ liệu được chia thành tập train/test.
3.  **Huấn luyện & Tuning:** Mô hình (RandomForest) được huấn luyện trên tập train. `GridSearchCV` được sử dụng để tìm siêu tham số tối ưu (tập trung vào Recall).
4.  **Đánh giá:** Mô hình tốt nhất được đánh giá trên tập test bằng nhiều chỉ số (Accuracy, Precision, Recall, F1, AUC). Kết quả được lưu và log vào MLflow.
5.  **Giải thích (XAI):** `SHAP` được sử dụng để tính toán độ ảnh hưởng của từng feature đến dự đoán của mô hình.
6.  **Đóng gói API:** Mô hình, scaler, và logic dự đoán/giải thích được đóng gói thành API bằng FastAPI.
7.  **Dockerization:** API được đóng gói vào Docker image.
8.  **CI/CD:** Khi code được push lên `master`:
    * GitHub Actions chạy lint, test.
    * Build Docker image mới.
    * Push image lên GHCR.
9.  **Deployment:** Render (qua Auto-Deploy) phát hiện image mới trên GHCR và tự động triển khai phiên bản API mới nhất.
10. **Sử dụng:** Người dùng tương tác với `frontend.html` (chạy cục bộ hoặc deploy riêng) hoặc các hệ thống khác gọi đến API đã triển khai trên Render để nhận dự đoán và giải thích.

## Hiệu năng Mô hình (Ví dụ)

Sau khi tinh chỉnh siêu tham số với GridSearchCV (tối ưu cho Recall), mô hình RandomForest đạt được các kết quả sau trên tập test:

* **Recall (Churn=1):** ~0.77 (Phát hiện được khoảng 77% khách hàng thực sự churn)
* **Precision (Churn=1):** ~0.54
* **F1-score (Churn=1):** ~0.64
* **ROC AUC:** ~0.84

*Lưu ý: Các chỉ số này có thể thay đổi một chút giữa các lần chạy hoặc nếu dữ liệu/tham số thay đổi.*



---
