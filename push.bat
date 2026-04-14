@echo off
setlocal

REM === CONFIG ===
set COMMIT_MSG=quick sync

echo Adding all changes...
git add .

echo Committing...
git commit -m "%COMMIT_MSG%" || echo Nothing to commit.

echo Pushing to remote...
git push

echo Done.
endlocal
pause
