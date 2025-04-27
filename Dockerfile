# Sử dụng image Python chính thức làm base image
# Chọn phiên bản Python phù hợp với môi trường ảo của bạn (ví dụ: 3.10, 3.11, ...)
FROM python:3.11-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Tối ưu hóa caching: Chỉ copy requirements.txt trước để cài đặt thư viện
# Nếu requirements.txt không đổi, bước này sẽ dùng cache khi build lại image
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
# --no-cache-dir để giảm kích thước image
# --upgrade pip để đảm bảo dùng pip mới nhất
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ nội dung dự án vào thư mục làm việc /app trong container
# Bao gồm thư mục src, models, config, api.py, ...
# Đảm bảo có file .dockerignore để loại bỏ các file không cần thiết (venv, __pycache__, ...)
COPY . .

# Mở cổng 8000 để API có thể được truy cập từ bên ngoài container
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI bằng Uvicorn khi container khởi động
# host 0.0.0.0 để cho phép truy cập từ bên ngoài container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

