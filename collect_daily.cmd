@echo off
rem Daily Naver research collection for Report Lens, run by Windows Task Scheduler.
rem Pages are followed until one holds only known reports (at most 30, about two
rem weeks), so days the PC was off are caught up on the next run.
rem Already downloaded reports are skipped, then target prices are extracted
rem into data\reports\consensus_cache.json so the consensus page opens quickly.
rem Set REPORT_LENS_PYTHON to use a specific interpreter (for example a venv).
setlocal
cd /d "%~dp0"
if "%REPORT_LENS_PYTHON%"=="" set "REPORT_LENS_PYTHON=python"
set "PYTHONIOENCODING=utf-8"
if not exist data mkdir data
echo ==== %date% %time% start >> data\collect_daily.log
"%REPORT_LENS_PYTHON%" main.py collect --save-dir data/reports --start-page 1 --end-page 30 --until-known >> data\collect_daily.log 2>&1
"%REPORT_LENS_PYTHON%" -c "import consensus; r, s = consensus.load_reports('data/reports'); print(f'consensus: {len(r)} reports, {len(s)} skipped')" >> data\collect_daily.log 2>&1
echo ==== %date% %time% end (exit %errorlevel%) >> data\collect_daily.log
endlocal
