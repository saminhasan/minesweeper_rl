@echo off
setlocal

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
	echo Error: this folder is not a git repository.
	endlocal
	pause
	exit /b 1
)

set "COMMIT_MSG=%~1"
if "%COMMIT_MSG%"=="" set "COMMIT_MSG=quick sync"

for /f "delims=" %%B in ('git branch --show-current') do set "BRANCH=%%B"
if "%BRANCH%"=="" set "BRANCH=main"

echo Current branch: %BRANCH%
echo.
echo Working tree status:
git status --short
echo.

echo Adding changes...
git add -A

git diff --cached --quiet
if not errorlevel 1 goto :commit

echo No staged changes to commit.
goto :push

:commit
echo Committing...
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
	echo Commit failed.
	endlocal
	pause
	exit /b 1
)

:push
echo Pushing to origin/%BRANCH%...
git push origin %BRANCH%
if errorlevel 1 (
	echo Push failed.
	endlocal
	pause
	exit /b 1
)

echo Done.
endlocal
pause
