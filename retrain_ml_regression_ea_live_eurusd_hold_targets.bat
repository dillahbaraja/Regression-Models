@echo off
setlocal

cd /d "%~dp0"

set "DATASET_DIR=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Regression_Datasets"
set "TRAIN_DATA=%DATASET_DIR%\EURUSD_H1_Data.csv"
set "TRAIN_NEWS_PREFERRED=%DATASET_DIR%\NEWS_2018.01.01-2023.12.31_ALL_.csv"
set "TRAIN_NEWS_FALLBACK=%DATASET_DIR%\NEWS_2018.01.01-2026.08.11_ALL_.csv"
set "TRAIN_NEWS=%TRAIN_NEWS_PREFERRED%"

if not exist "%TRAIN_NEWS%" (
    set "TRAIN_NEWS=%TRAIN_NEWS_FALLBACK%"
)

echo [1/7] Checking required files...
if not exist "%TRAIN_DATA%" (
    echo Missing EURUSD OHLCV file:
    echo %TRAIN_DATA%
    pause
    exit /b 1
)
if not exist "%TRAIN_NEWS%" (
    echo Missing training news file. Checked:
    echo %TRAIN_NEWS_PREFERRED%
    echo %TRAIN_NEWS_FALLBACK%
    pause
    exit /b 1
)

if /i "%TRAIN_NEWS%"=="%TRAIN_NEWS_FALLBACK%" (
    echo WARNING:
    echo Preferred train-only news file was not found.
    echo This batch will use:
    echo %TRAIN_NEWS_FALLBACK%
    echo That file includes dates beyond 2023 and is not a clean train-only news source.
    echo.
)

echo [2/7] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found in PATH.
    pause
    exit /b 1
)

echo [3/7] Installing or updating required packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo [4/7] Cleaning EURUSD standard assets only...
if not exist "Assets" mkdir "Assets"
if not exist "%APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets" mkdir "%APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets"

del /q "Assets\EURUSD_H1_*" 2>nul
del /q "%APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets\EURUSD_H1_*" 2>nul
del /q "models\trained\EURUSD_H1_*" 2>nul
del /q "models\params\EURUSD_H1_*" 2>nul
del /q "reports\metrics\EURUSD_H1_*" 2>nul
del /q "reports\charts\EURUSD_H1_*" 2>nul
del /q "reports\tuning\EURUSD_H1_*" 2>nul
del /q "data\processed\EURUSD_H1_*" 2>nul

echo [5/7] Training EURUSD with model-side blacklist suppression...
echo News file:
echo %TRAIN_NEWS%
echo.

python src\train_model.py ^
  --symbols EURUSD ^
  --models xgboost lightgbm catboost sklearn_hgb voting ^
  --train-all-data ^
  --data-csv "%TRAIN_DATA%" ^
  --news-csv "%TRAIN_NEWS%" ^
  --no-runtime-export ^
  --suppress-blacklist-events ^
  --blacklist-target-factor 0.0

if errorlevel 1 (
    echo Training failed for EURUSD.
    pause
    exit /b 1
)

echo [6/7] EURUSD retraining with hold-target suppression completed successfully.
echo Output updated in:
echo - Assets
echo - %APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets
echo.
echo [7/7] Done.
pause

endlocal
