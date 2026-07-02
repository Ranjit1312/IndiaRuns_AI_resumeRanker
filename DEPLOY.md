# Deploy — GitHub + Hugging Face Space (v1)

Run from the project root in **PowerShell**. The repo is already committed.

## 0. (optional) Test locally first
```powershell
./.venv/Scripts/python -m streamlit run app.py
# open the shown URL, paste your Google AI Studio key in the sidebar, paste a JD
# (e.g. the text from data/eval_jds/nvidia_solutions-architect-ai-ml/jd.md), Compile.
```

## 1. Push to a new GitHub repo
Create an **empty** repo at github.com/new (no README), then:
```powershell
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main          # Git Credential Manager handles auth (browser popup)
```

## 2. Create + deploy the Hugging Face Space
Create a Space at huggingface.co/new-space → **SDK: Streamlit**, hardware **CPU basic** (free).
The `README.md` already has the required Space header (`sdk: streamlit`, `app_file: app.py`).
Then push this repo to the Space's git remote:
```powershell
git remote add space https://huggingface.co/spaces/<USER>/<SPACE>.git
git push space main
# HF auth: your HF username + a WRITE access token (huggingface.co/settings/tokens)
```
The Space builds from `requirements.txt` (light — no torch), then serves `app.py`.

**BYO key:** the Space ships with NO key. Each user pastes their own Google AI Studio
key in the sidebar (session-only). Do NOT add GOOGLE_API_KEY as a Space secret unless
you intend to pay for everyone's usage.

## 3. (optional) Run the multi-model parity eval vs the gold set
```powershell
$env:GOOGLE_API_KEY = "<your-key>"
./.venv/Scripts/python -m harness.parity --eval-dir data/eval_jds --models "gemma-4-26b-a4b-it,gemma-4-31b-it"
# or compile one JD:
./.venv/Scripts/python -m harness.coerce --jd data/eval_jds/stripe_backend-software-engineer/jd.txt --model gemma-4-26b-a4b-it
```

## Notes
- If `git push space` rejects large files, confirm `.venv/` and `.env` are gitignored
  (they are) — nothing large should be tracked.
- The Space only needs `app.py`, `harness/`, `jd/`, `redrob_ranker/profile.py`,
  `requirements.txt`, `README.md`. `data/eval_jds/` ships too (small) for the parity tab / demos.
