@echo off
echo ============================================
echo Starting Patient Data Extraction OCR System
echo ============================================
echo.

echo Starting Backend Server on http://localhost:8000
start "Backend Server" cmd /k "cd backend && python app.py"

timeout /t 3 /nobreak > nul

echo Starting Frontend Server on http://localhost:3000
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ============================================
echo System is starting...
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo ============================================
echo.
echo Press any key to stop all servers...
pause > nul

taskkill /FI "WindowTitle eq Backend Server*" /T /F
taskkill /FI "WindowTitle eq Frontend Server*" /T /F

