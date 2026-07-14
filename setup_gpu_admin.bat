@echo off
REM ============================================================
REM  GPU Performance Setup - Run as Administrator
REM
REM  This script configures NVIDIA driver and Windows power
REM  settings to prevent GPU throttling on battery power.
REM
REM  Run: Right-click -> Run as administrator
REM  Or:  Open admin PowerShell and run this .bat file
REM ============================================================

echo.
echo ============================================================
echo  GPU Performance Setup (requires Administrator)
echo ============================================================
echo.

REM Check for admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [1/6] Setting NVIDIA PerfLevelSrc (force max performance)...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001" /v "PerfLevelSrc" /t REG_DWORD /d 0x33222220 /f
echo.

echo [2/6] Setting NVIDIA PowerMizerEnable (disable power saving)...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001" /v "PowerMizerEnable" /t REG_DWORD /d 0x00000000 /f
echo.

echo [3/6] Setting NVIDIA PowerMizerLevel (max performance on AC)...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001" /v "PowerMizerLevel" /t REG_DWORD /d 0x00000001 /f
echo.

echo [4/6] Setting NVIDIA PowerMizerLevelDC (max performance on battery)...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001" /v "PowerMizerLevelDC" /t REG_DWORD /d 0x00000001 /f
echo.

echo [5/6] Disabling PCI Express Link State Power Management...
powercfg /setacvalueindex SCHEME_CURRENT 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0
powercfg /setdcvalueindex SCHEME_CURRENT 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0
powercfg /setactive SCHEME_CURRENT
echo Done.
echo.

echo [6/6] Setting processor max state to 100%%...
powercfg /setacvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 100
powercfg /setdcvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 100
powercfg /setactive SCHEME_CURRENT
echo Done.
echo.

echo ============================================================
echo  Setup complete! Changes take effect immediately.
echo  You can now run 'python main.py' on battery with full GPU.
echo ============================================================
echo.
pause
