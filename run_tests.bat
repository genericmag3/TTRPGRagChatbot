@echo off
setlocal

echo ============================================================
echo  Running tests with coverage
echo ============================================================
echo.

python3 -m pytest tests/ --cov --cov-report=term-missing --cov-report=html -v

set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% NEQ 0 (
    echo ============================================================
    echo  Tests FAILED  ^(exit code %EXIT_CODE%^)
    echo ============================================================
    exit /b %EXIT_CODE%
)

echo ============================================================
echo  All tests passed
echo  HTML report: tests\coverage_report\index.html
echo ============================================================

start "" "tests\coverage_report\index.html"

endlocal
