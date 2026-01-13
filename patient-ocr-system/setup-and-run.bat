@echo off
echo ============================================
echo Patient Data Extraction OCR System Setup
echo ============================================
echo.

echo [1/4] Setting up Backend...
cd backend
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)
echo Backend setup complete!
echo.

echo [2/4] Setting up Frontend...
cd ..\frontend
echo Installing Node.js dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)
echo Frontend setup complete!
echo.

echo [3/4] Creating sample patient document...
cd ..
echo Please open sample-patient-document.html in your browser and take a screenshot to test the system.
echo.

echo [4/4] Setup Complete!
echo.
echo ============================================
echo IMPORTANT: Make sure Tesseract OCR is installed
echo Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo Add Tesseract to PATH or set pytesseract.pytesseract.tesseract_cmd in app.py
echo ============================================
echo.
echo To run the system:
echo   Backend:  cd backend ^&^& python app.py
echo   Frontend: cd frontend ^&^& npm start
echo.
echo Or run run-system.bat to start both servers
echo.
pause

