@echo OFF
ECHO Activating virtual environment...
SET PROJECT_DIR=D:\churn_prediction_project

SET VENV_ACTIVATE=%PROJECT_DIR%\venv\Scripts\activate.bat

REM Kích hoạt môi trường ảo
CALL "%VENV_ACTIVATE%"

REM Kiểm tra xem kích hoạt venv thành công không (tùy chọn)
IF "%VIRTUAL_ENV%"=="" (
    ECHO ERROR: Failed to activate virtual environment. Check VENV_ACTIVATE path.
    PAUSE
    EXIT /B 1
)

ECHO Changing directory to project folder...
cd /D "%PROJECT_DIR%"
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Failed to change directory to %PROJECT_DIR%. Check PROJECT_DIR path.
    PAUSE
    EXIT /B 1
)


ECHO Starting Python training pipeline (stage: full)...
REM Chạy stage 'full' để đảm bảo dữ liệu được xử lý và model được train lại
REM Chuyển hướng output và lỗi vào file log (tùy chọn nhưng rất hữu ích)
python main.py full >> pipeline.log 2>&1

IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Python script main.py failed. Check pipeline.log for details.
    PAUSE
    EXIT /B 1
) ELSE (
    ECHO Python script main.py completed successfully. Check pipeline.log for output.
)


ECHO Pipeline finished. Deactivating environment (optional)...
REM CALL deactivate

ECHO Automation script finished.
REM PAUSE
