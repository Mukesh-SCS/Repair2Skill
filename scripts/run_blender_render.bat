@echo off
REM Set paths without quotes
set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.5\blender.exe
set SCRIPT_PATH=C:\Users\tripa\Downloads\GitHub_Projects\Repair2Skill\scripts\blender_damage_simulation.py

for /l %%i in (1,1,20) do (
    echo Rendering image %%i...
    "%BLENDER_PATH%" --background --python "%SCRIPT_PATH%"
)

pause
