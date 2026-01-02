@echo off
cd /d "%~dp0"
title BALLS! (v18) Launcher

echo Checking for Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed or not in your PATH.
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b
)

echo.
echo Launching BALLS! (v18)...
echo This may take a few minutes the first time to build the container.
echo.
docker-compose up --build

echo.
echo Application stopped.
pause
