@echo off
setlocal enabledelayedexpansion

for %%D in (
  MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
  V1_01_easy V1_03_difficult V2_01_easy V2_03_difficult
) do (
  for %%O in (1 5 10 15 20 30 40) do (
    echo Запуск %%D offset=%%O
    python main.py --path "..\Datasets\%%D" --offset %%O
  )
)

endlocal
