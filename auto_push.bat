@echo off
cd /d "C:\Users\ahmet.basaran\Desktop\PROJELER\checkerboard-corner-detector"
git add -A
git diff --cached --quiet && echo Nothing to commit. || git commit -m "Auto-push: %date:~10,4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%"
git push
