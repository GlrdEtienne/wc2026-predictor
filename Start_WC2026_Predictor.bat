@echo off
call conda activate wc2026
set PYTHONIOENCODING=utf-8

echo Mise a jour des scores...
python src/collect/update_live.py

echo Lancement du dashboard...
streamlit run src/dashboard/app.py