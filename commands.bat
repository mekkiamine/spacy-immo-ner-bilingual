@echo off
REM commands.bat - Script d'installation Windows

echo ========================================
echo    NLP-Urbanova Setup Script (Windows)
echo ========================================
echo.

REM Verifier Python
echo Verification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe!
    pause
    exit /b 1
)
echo [OK] Python trouve

REM Creer environnement virtuel
echo.
echo Creation de l'environnement virtuel...
if not exist .venv (
    python -m venv .venv
    echo [OK] Environnement virtuel cree
) else (
    echo [INFO] Environnement virtuel existe deja
)

REM Activer environnement
echo.
echo Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat
echo [OK] Environnement active

REM Installer dependances
echo.
echo Installation des dependances...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo [OK] Dependances installees

REM Menu
echo.
echo ========================================
echo Que voulez-vous faire?
echo ========================================
echo 1) Entrainer le modele (20-40 min)
echo 2) Tester le modele
echo 3) Test rapide avec une phrase
echo 4) Voir les statistiques
echo 5) Quitter
echo.
set /p choice="Votre choix (1-5): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto test
if "%choice%"=="3" goto quick_test
if "%choice%"=="4" goto stats
if "%choice%"=="5" goto end
goto invalid

:train
echo.
echo [INFO] Entrainement du modele...
echo ========================================
python 1_annotate_data.py
if %errorlevel% neq 0 goto error
python 2_train_model.py
if %errorlevel% neq 0 goto error
python -m spacy train config_bilingual_fixed.cfg --output output_model_immo_ner_bilingual_v3 --paths.train train_bilingual_V3.spacy --paths.dev dev_bilingual_V3.spacy
if %errorlevel% neq 0 goto error
echo [OK] Modele entraine avec succes!
goto end

:test
echo.
echo [INFO] Test du modele...
python 3_test_model.py
echo [OK] Tests termines! Consultez test_results/index.html
goto end

:quick_test
echo.
echo [INFO] Test rapide...
python -c "import spacy; nlp=spacy.load('output_model_immo_ner_bilingual_v3/model-best'); doc=nlp('Appartement 3 chambres a louer Tunis 800 TND'); [print(f'{ent.text} -> {ent.label_}') for ent in doc.ents]"
goto end

:stats
echo.
echo [INFO] Statistiques du dataset...
python -c "import pandas as pd; df=pd.read_csv('house_price_bd.csv'); print(f'\nNombre d annonces: {len(df)}\n'); print(df.head())"
goto end

:error
echo [ERREUR] Une erreur s'est produite!
pause
exit /b 1

:invalid
echo [ERREUR] Choix invalide!
pause
exit /b 1

:end
echo.
echo [OK] Termine!
pause