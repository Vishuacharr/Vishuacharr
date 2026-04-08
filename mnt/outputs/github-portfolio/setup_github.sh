#!/usr/bin/env bash
# =============================================================================
# GitHub Portfolio Setup Script — Vishruth Acharya (@Vishuacharr)
# =============================================================================
# Prerequisites:
#   1. Install GitHub CLI:  https://cli.github.com/
#   2. Authenticate:        gh auth login
#   3. Run this script:     bash setup_github.sh
# =============================================================================

set -euo pipefail

GITHUB_USER="Vishuacharr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color helpers
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*"; exit 1; }
section() { echo -e "\n${YELLOW}══════════════════════════════════════${NC}"; echo -e "${YELLOW} $*${NC}"; echo -e "${YELLOW}══════════════════════════════════════${NC}"; }

# Check prerequisites
command -v gh  &>/dev/null || error "GitHub CLI not found. Install: https://cli.github.com/"
command -v git &>/dev/null || error "git not found."
gh auth status &>/dev/null || error "Not authenticated. Run: gh auth login"

# =============================================================================
# 1. Profile README repo (Vishuacharr/Vishuacharr)
# =============================================================================
section "Step 1: Creating Profile README"

PROFILE_REPO="${GITHUB_USER}/${GITHUB_USER}"
if gh repo view "${PROFILE_REPO}" &>/dev/null; then
    warn "Profile repo already exists — updating README..."
    TMP=$(mktemp -d)
    gh repo clone "${PROFILE_REPO}" "${TMP}" -- --depth 1
    cp "${SCRIPT_DIR}/profile-readme/README.md" "${TMP}/README.md"
    cd "${TMP}"
    git add README.md
    git diff --cached --quiet || git commit -m "feat: update profile README with AI portfolio"
    git push
    cd "${SCRIPT_DIR}"
    rm -rf "${TMP}"
else
    gh repo create "${PROFILE_REPO}" --public --description "Vishruth Acharya — AI Engineer | RAG | YOLO | LLMs" 2>/dev/null || true
    TMP=$(mktemp -d)
    cd "${TMP}"
    git init
    cp "${SCRIPT_DIR}/profile-readme/README.md" ./README.md
    git add .
    git commit -m "feat: add profile README"
    git branch -M main
    git remote add origin "https://github.com/${PROFILE_REPO}.git"
    git push -u origin main
    cd "${SCRIPT_DIR}"
    rm -rf "${TMP}"
fi
info "Profile README done!"

# =============================================================================
# Helper: create_and_push <folder> <repo-name> <description> <topics>
# =============================================================================
create_and_push() {
    local folder="$1"
    local repo="$2"
    local desc="$3"
    local topics="$4"
    local full_repo="${GITHUB_USER}/${repo}"

    section "Creating: ${repo}"

    # Create repo if not exists
    if ! gh repo view "${full_repo}" &>/dev/null; then
        gh repo create "${full_repo}" \
            --public \
            --description "${desc}" \
            --clone=false 2>/dev/null || warn "Repo may already exist"
    fi

    # Add topics
    gh repo edit "${full_repo}" --add-topic "${topics}" 2>/dev/null || true

    # Init and push
    cd "${SCRIPT_DIR}/${folder}"
    if [ ! -d ".git" ]; then
        git init
        git branch -M main
        git remote add origin "https://github.com/${full_repo}.git"
    fi

    # Add .gitignore if missing
    [ -f ".gitignore" ] || cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
*.log
*.pkl
*.pt
*.bin
models/
data/raw/
.DS_Store
GITIGNORE

    git add -A
    git diff --cached --quiet || git commit -m "feat: initial commit — ${repo}"
    git push -u origin main --force 2>/dev/null || git push -u origin main

    cd "${SCRIPT_DIR}"
    info "✅ ${repo} pushed to https://github.com/${full_repo}"
}

# =============================================================================
# 2. RAG Document Intelligence
# =============================================================================
create_and_push \
    "rag-document-intelligence" \
    "rag-document-intelligence" \
    "Production RAG pipeline: LangChain + LangGraph multi-agent + pgvector + RAGAS eval (faithfulness: 0.91)" \
    "rag,langchain,langgraph,pgvector,llm,python,fastapi,ai,nlp,retrieval-augmented-generation"

# =============================================================================
# 3. YOLO Vision AI
# =============================================================================
create_and_push \
    "yolo-vision-ai" \
    "yolo-vision-ai" \
    "Real-time object detection with YOLOv8: FastAPI inference server + Streamlit UI + Docker" \
    "yolo,yolov8,object-detection,computer-vision,fastapi,streamlit,opencv,python,deep-learning"

# =============================================================================
# 4. UTD AI Chatbot
# =============================================================================
create_and_push \
    "utd-ai-chatbot" \
    "utd-ai-chatbot" \
    "Llama 3.2 3B fine-tuned with QLoRA on UTD academic data — 40% accuracy improvement, 5000+ monthly users" \
    "llm,llama,qlora,fine-tuning,peft,chatbot,transformers,nlp,python,ai"

# =============================================================================
# 5. Data Lakehouse ETL
# =============================================================================
create_and_push \
    "data-lakehouse-etl" \
    "data-lakehouse-etl" \
    "Medallion architecture (Bronze→Silver→Gold) with PySpark, Delta Lake, Kafka CDC, and data quality checks" \
    "pyspark,delta-lake,kafka,data-engineering,etl,lakehouse,aws,databricks,python,streaming"

# =============================================================================
# 6. Pin the best repos
# =============================================================================
section "Step 6: Pinning top repositories"
cat << 'EOF'
⚠️  GitHub doesn't support pinning repos via CLI yet.
   Manually pin these 4 repos on your profile:
   → https://github.com/Vishuacharr

   1. rag-document-intelligence
   2. yolo-vision-ai
   3. utd-ai-chatbot
   4. data-lakehouse-etl

EOF

# =============================================================================
# Done!
# =============================================================================
section "🎉 GitHub Portfolio Setup Complete!"
cat << EOF
Your profile: https://github.com/${GITHUB_USER}

Repos created:
  ✅ ${GITHUB_USER}/${GITHUB_USER}          — Profile README
  ✅ rag-document-intelligence              — RAG + LangGraph + pgvector
  ✅ yolo-vision-ai                         — YOLOv8 + FastAPI + Streamlit
  ✅ utd-ai-chatbot                         — Llama 3.2 + QLoRA fine-tuning
  ✅ data-lakehouse-etl                     — PySpark + Delta Lake + Kafka

Next steps:
  1. Pin your top 4 repos on your GitHub profile page
  2. Update LinkedIn with your GitHub profile URL
  3. Add a profile picture and bio at https://github.com/settings/profile
  4. Star your own repos to show activity

EOF
