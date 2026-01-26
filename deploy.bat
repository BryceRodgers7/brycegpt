@echo off
REM BryceGPT Deployment Script for Google Cloud Run (Windows)
REM This script automates the deployment process

echo.
echo ================================
echo BryceGPT Deployment Script
echo ================================
echo.

REM Check if gcloud is installed
where gcloud >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: gcloud CLI is not installed
    echo Please install it from: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Get project ID
for /f "delims=" %%i in ('gcloud config get-value project 2^>nul') do set PROJECT_ID=%%i

if "%PROJECT_ID%"=="" (
    echo Error: No Google Cloud project configured
    echo Please run: gcloud config set project YOUR_PROJECT_ID
    exit /b 1
)

echo Project ID: %PROJECT_ID%
echo.

REM Configuration
set SERVICE_NAME=brycegpt
set REGION=us-central1
set MEMORY=2Gi
set CPU=2
set TIMEOUT=300

echo Configuration:
echo    Service Name: %SERVICE_NAME%
echo    Region: %REGION%
echo    Memory: %MEMORY%
echo    CPU: %CPU%
echo    Timeout: %TIMEOUT%s
echo.

REM Confirm deployment
set /p CONTINUE="Continue with deployment? (y/n): "
if /i not "%CONTINUE%"=="y" (
    echo Deployment cancelled
    exit /b 0
)

echo.
echo Building container image...
gcloud builds submit --tag gcr.io/%PROJECT_ID%/%SERVICE_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo Build failed
    exit /b 1
)

echo.
echo Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% --image gcr.io/%PROJECT_ID%/%SERVICE_NAME% --platform managed --region %REGION% --allow-unauthenticated --memory %MEMORY% --cpu %CPU% --timeout %TIMEOUT%
if %ERRORLEVEL% NEQ 0 (
    echo Deployment failed
    exit /b 1
)

REM Get the service URL
for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo.
echo ================================
echo Deployment complete!
echo ================================
echo.
echo Service URL: %SERVICE_URL%
echo.
echo API Endpoints:
echo    Health Check: %SERVICE_URL%/health
echo    Generate: %SERVICE_URL%/generate
echo    Vocabulary: %SERVICE_URL%/vocab
echo    Docs: %SERVICE_URL%/docs
echo.
echo Test your deployment:
echo    curl %SERVICE_URL%/health
echo.
echo Update your frontend with this URL:
echo    API_URL = "%SERVICE_URL%"
echo.

