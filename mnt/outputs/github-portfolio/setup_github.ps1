# =============================================================================
# GitHub Portfolio Setup — Vishruth Acharya (@Vishuacharr)
# PowerShell version for Windows
# =============================================================================
# Run this in a NEW PowerShell window after installing GitHub CLI:
#   winget install --id GitHub.cli -e
#   gh auth login
#   .\setup_github.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$GITHUB_USER = "Vishuacharr"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Green  { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Yellow { param($msg) Write-Host "[>>] $msg" -ForegroundColor Yellow }
function Write-Section { param($msg) Write-Host "`n====== $msg ======" -ForegroundColor Cyan }

# Check gh is available
try { gh --version | Out-Null } catch { Write-Host "ERROR: gh not found. Run: winget install --id GitHub.cli -e" -ForegroundColor Red; exit 1 }
try { gh auth status 2>&1 | Out-Null } catch { Write-Host "ERROR: Not logged in. Run: gh auth login" -ForegroundColor Red; exit 1 }

# Helper: create repo + push folder
function Push-Repo {
    param(
        [string]$Folder,
        [string]$RepoName,
        [string]$Description,
        [string[]]$Topics
    )

    Write-Section "Creating: $RepoName"
    $fullRepo = "$GITHUB_USER/$RepoName"
    $localPath = Join-Path $SCRIPT_DIR $Folder

    # Create repo (ignore if exists)
    $repoExists = gh repo view $fullRepo 2>&1
    if ($LASTEXITCODE -ne 0) {
        gh repo create $fullRepo --public --description $Description | Out-Null
        Write-Yellow "Repo created: $fullRepo"
    } else {
        Write-Yellow "Repo already exists, updating..."
    }

    # Add topics
    foreach ($t in $Topics) {
        gh repo edit $fullRepo --add-topic $t 2>&1 | Out-Null
    }

    # Git init + push
    Push-Location $localPath

    if (-not (Test-Path ".git")) {
        git init | Out-Null
        git branch -M main | Out-Null
        git remote add origin "https://github.com/$fullRepo.git" | Out-Null
    }

    # Create .gitignore if missing
    if (-not (Test-Path ".gitignore")) {
        @"
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
*.egg-info/
*.log
*.pkl
*.pt
*.bin
*.safetensors
data/raw/
.DS_Store
"@ | Set-Content ".gitignore"
    }

    git add -A | Out-Null
    $status = git status --porcelain
    if ($status) {
        git commit -m "feat: initial commit — $RepoName" | Out-Null
    }
    git push -u origin main --force 2>&1 | Out-Null

    Pop-Location
    Write-Green "$RepoName pushed to https://github.com/$fullRepo"
}

# =============================================================================
# 1. Profile README  (Vishuacharr/Vishuacharr)
# =============================================================================
Write-Section "Step 1: Profile README"
$profileRepo = "$GITHUB_USER/$GITHUB_USER"
$profileExists = gh repo view $profileRepo 2>&1
if ($LASTEXITCODE -ne 0) {
    gh repo create $profileRepo --public --description "Vishruth Acharya — AI Engineer" | Out-Null
}
$tmpDir = Join-Path $env:TEMP "vishu_profile_$(Get-Random)"
New-Item -ItemType Directory -Path $tmpDir | Out-Null
Push-Location $tmpDir
git init | Out-Null
git branch -M main | Out-Null
Copy-Item (Join-Path $SCRIPT_DIR "profile-readme\README.md") ".\README.md"
git add . | Out-Null
git commit -m "feat: add profile README" | Out-Null
git remote add origin "https://github.com/$profileRepo.git" | Out-Null
git push -u origin main --force 2>&1 | Out-Null
Pop-Location
Remove-Item -Recurse -Force $tmpDir
Write-Green "Profile README live at https://github.com/$GITHUB_USER"

# =============================================================================
# 2-5. AI Project Repos
# =============================================================================
Push-Repo `
    -Folder "rag-document-intelligence" `
    -RepoName "rag-document-intelligence" `
    -Description "Production RAG pipeline: LangChain + LangGraph multi-agent + pgvector + RAGAS eval (faithfulness: 0.91)" `
    -Topics @("rag","langchain","langgraph","pgvector","llm","python","fastapi","ai","nlp")

Push-Repo `
    -Folder "yolo-vision-ai" `
    -RepoName "yolo-vision-ai" `
    -Description "Real-time object detection with YOLOv8: FastAPI inference server + Streamlit UI + Docker" `
    -Topics @("yolo","yolov8","object-detection","computer-vision","fastapi","streamlit","opencv","python")

Push-Repo `
    -Folder "utd-ai-chatbot" `
    -RepoName "utd-ai-chatbot" `
    -Description "Llama 3.2 3B fine-tuned with QLoRA on UTD academic data — 40% accuracy improvement, 5000+ monthly users" `
    -Topics @("llm","llama","qlora","fine-tuning","peft","chatbot","transformers","nlp","python")

Push-Repo `
    -Folder "data-lakehouse-etl" `
    -RepoName "data-lakehouse-etl" `
    -Description "Medallion architecture (Bronze to Silver to Gold) with PySpark, Delta Lake, Kafka CDC, and data quality checks" `
    -Topics @("pyspark","delta-lake","kafka","data-engineering","etl","lakehouse","databricks","python")

# =============================================================================
# Done!
# =============================================================================
Write-Section "All Done!"
Write-Host @"

Your GitHub profile: https://github.com/$GITHUB_USER

Repos created:
  [OK] $GITHUB_USER/$GITHUB_USER          — Profile README
  [OK] rag-document-intelligence          — RAG + LangGraph + pgvector
  [OK] yolo-vision-ai                     — YOLOv8 + FastAPI + Streamlit
  [OK] utd-ai-chatbot                     — Llama 3.2 + QLoRA fine-tuning
  [OK] data-lakehouse-etl                 — PySpark + Delta Lake + Kafka

NEXT STEPS (takes 2 minutes):
  1. Go to https://github.com/$GITHUB_USER
  2. Click 'Customize your pins'
  3. Pin: rag-document-intelligence, yolo-vision-ai, utd-ai-chatbot, data-lakehouse-etl
  4. Add a profile photo + bio at https://github.com/settings/profile

"@ -ForegroundColor Green
