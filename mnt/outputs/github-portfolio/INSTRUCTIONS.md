# 🚀 GitHub Portfolio Setup — Vishruth Acharya

## What's Included

| Folder | GitHub Repo | Description |
|--------|------------|-------------|
| `profile-readme/` | `Vishuacharr/Vishuacharr` | Animated profile README |
| `rag-document-intelligence/` | `rag-document-intelligence` | RAG + LangGraph + pgvector |
| `yolo-vision-ai/` | `yolo-vision-ai` | YOLOv8 detection + FastAPI |
| `utd-ai-chatbot/` | `utd-ai-chatbot` | Llama 3.2 + QLoRA fine-tuning |
| `data-lakehouse-etl/` | `data-lakehouse-etl` | PySpark Bronze→Silver→Gold |

---

## ⚡ One-Command Setup (Recommended)

### Step 1: Install GitHub CLI
```bash
# macOS
brew install gh

# Windows
winget install GitHub.cli

# Linux
sudo apt install gh
```

### Step 2: Authenticate
```bash
gh auth login
# → Choose GitHub.com → HTTPS → Login with browser
```

### Step 3: Run the setup script
```bash
bash setup_github.sh
```

That's it! All repos are created, populated, and pushed.

---

## 📌 Manual Pinning (Do This After!)

1. Go to https://github.com/Vishuacharr
2. Click **"Customize your pins"**
3. Pin these 4 repos:
   - `rag-document-intelligence`
   - `yolo-vision-ai`
   - `utd-ai-chatbot`
   - `data-lakehouse-etl`

---

## 👤 Profile Polish Checklist

- [ ] Add a professional headshot at https://github.com/settings/profile
- [ ] Add bio: *"AI Engineer | MS Business Analytics & AI @ UTD | RAG · YOLO · LLMs · Data Engineering"*
- [ ] Add location: *Dallas, TX*
- [ ] Add LinkedIn URL
- [ ] Pin top 4 repos (see above)
- [ ] Enable contribution graph (settings → make private contributions visible)

---

## 🔑 Adding API Keys to Repos

For the RAG and Chatbot projects, create a `.env` file (never commit it!):

```bash
# rag-document-intelligence/.env
OPENAI_API_KEY=sk-...
PGVECTOR_URL=postgresql://user:pass@localhost:5432/ragdb

# utd-ai-chatbot/.env
HF_TOKEN=hf_...
```

---

## 📧 Questions?
Contact: vishruthacharr@gmail.com
