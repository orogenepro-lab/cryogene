@echo off
REM ===== CUVETTE : prevision quotidienne (a lancer par le Planificateur de taches) =====
REM Prend automatiquement le dernier run IFS complet (00 ou 12 UTC), genere la page et
REM l'enregistre dans C:\cuvette_archive\prevision\cuvette_derniere.html
call C:\Users\nikle\anaconda3\Scripts\activate.bat icon
cd /d C:\Users\nikle\Documents\THUNDER_TOOL\V2
echo ===== %date% %time% ===== >> C:\cuvette_archive\cuvette_log.txt
python forecast_html.py --no-open >> C:\cuvette_archive\cuvette_log.txt 2>&1
