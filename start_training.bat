@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "DATASET_DIR=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Regression_Datasets"
set "TRAIN_NEWS_PREFERRED=%DATASET_DIR%\NEWS_2018.01.01-2023.12.31_ALL_.csv"
set "TRAIN_NEWS_FALLBACK=%DATASET_DIR%\NEWS_2018.01.01-2026.08.11_ALL_.csv"
set "TRAIN_NEWS=%TRAIN_NEWS_PREFERRED%"

if not exist "%TRAIN_NEWS%" (
    set "TRAIN_NEWS=%TRAIN_NEWS_FALLBACK%"
)

echo [1/7] Checking dataset inputs...
if not exist "%DATASET_DIR%" (
    echo Missing dataset folder:
    echo %DATASET_DIR%
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
    echo.
)

echo News file:
echo %TRAIN_NEWS%
echo Active pairs:
echo - AUDUSD
echo - EURUSD
echo - GBPUSD
echo - USDJPY
echo.
echo Excluded pairs:
echo - USDCHF  ^(no valid CH coverage in current news source^)
echo - USDCAD  ^(no valid CA coverage in current news source^)
echo.

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

echo [4/7] Cleaning active-pair standard artifacts only...
if not exist "Assets" mkdir "Assets"
if not exist "%APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets" mkdir "%APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets"
if not exist "models\trained" mkdir "models\trained"
if not exist "models\params" mkdir "models\params"
if not exist "reports\metrics" mkdir "reports\metrics"
if not exist "reports\charts" mkdir "reports\charts"
if not exist "reports\tuning" mkdir "reports\tuning"
if not exist "data\processed" mkdir "data\processed"

for %%S in (AUDUSD EURUSD GBPUSD USDJPY) do (
    del /q "Assets\%%S_H1_*" 2>nul
    del /q "%APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets\%%S_H1_*" 2>nul
    del /q "models\trained\%%S_H1_*" 2>nul
    del /q "models\params\%%S_H1_*" 2>nul
    del /q "reports\metrics\%%S_H1_*" 2>nul
    del /q "reports\charts\%%S_H1_*" 2>nul
    del /q "reports\tuning\%%S_H1_*" 2>nul
    del /q "data\processed\%%S_H1_*" 2>nul
)

echo [5/7] Training active pairs with model-side blacklist suppression...
echo.

for %%S in (AUDUSD EURUSD GBPUSD USDJPY) do (
    set "TRAIN_DATA=%DATASET_DIR%\%%S_H1_Data.csv"
    if not exist "!TRAIN_DATA!" (
        echo Missing OHLCV file:
        echo !TRAIN_DATA!
        pause
        exit /b 1
    )

    echo ============================================================
    echo Training %%S from:
    echo !TRAIN_DATA!
    echo ============================================================

    python src\train_model.py ^
      --symbols %%S ^
      --models xgboost lightgbm catboost sklearn_hgb voting ^
      --train-all-data ^
      --data-csv "!TRAIN_DATA!" ^
      --news-csv "%TRAIN_NEWS%" ^
      --no-runtime-export ^
      --suppress-blacklist-events ^
      --blacklist-target-factor 0.0

    if errorlevel 1 (
        echo.
        echo Training failed for %%S.
        pause
        exit /b 1
    )

    echo Completed %%S.
    echo.
)

echo [6/7] Active-pair retraining completed successfully.
echo Output updated in:
echo - Assets
echo - %APPDATA%\MetaQuotes\Terminal\Common\Files\Regression_Assets
echo.
echo [7/7] Done.
pause

endlocal
