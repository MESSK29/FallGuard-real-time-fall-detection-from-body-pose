# FallGuard: Privacy-Preserving Fall Detection System

> [!WARNING]
> **DISCLAIMER: NOT A MEDICAL DEVICE**  
> FallGuard is an educational student research and demonstration prototype. It is **not** a certified medical device, healthcare diagnostic tool, or replacement for human care and certified commercial emergency response systems.

FallGuard is an edge-friendly, camera-based fall detection system tailored for elderly care environments. It leverages pose estimation landmarks to detect falls with strict privacy safeguards.

---

## Key Design Principles

1. **Strict Privacy**: Video stays in the browser unless you switch on incident clips. When enabled, a 20-second clip without audio is sent only after a confirmed fall, to the caregiver's Telegram chat, and is deleted from the server after sending. Telegram bot chats are not end-to-end encrypted. Otherwise, only 33 MediaPipe pose landmark coordinates $(x, y, z, \text{visibility})$ and derived kinematic features are processed.
2. **CPU-Only Efficiency**: Optimized to execute on standard consumer laptop CPUs without requiring dedicated GPUs.
3. **Single Source of Truth (`src/features.py`)**: A single shared module implements both per-frame kinematic metrics and 2-second sliding window summary statistics, guaranteeing zero train-test-inference feature drift.
4. **Two-Stage Confirmation**: Prevents false alarms by requiring a classifier trigger (Random Forest or XGBoost) followed by a ~2.0-second lying-still verification window (horizontal torso posture + near-zero velocity).
5. **No Data Leakage**: Video-grouped cross-validation (`GroupKFold` strictly by video ID) ensures temporal sliding windows from the same recording are never split across train and validation sets.

---

## Project Structure

```text
fallguard/
├── data/
│   ├── raw/             # Raw input video files (git-ignored)
│   ├── landmarks/       # Extracted pose landmark files (git-ignored)
│   └── features/        # Precomputed sliding-window feature sets (git-ignored)
├── notebooks/           # Exploratory data analysis & feature exploration notebooks
├── src/
│   ├── __init__.py
│   ├── extract_landmarks.py  # MediaPipe pose extraction from video clips
│   ├── features.py           # Shared per-frame & sliding-window feature extraction
│   ├── train.py              # ML model training (RF & XGBoost) with GroupKFold
│   ├── baseline_rules.py     # Heuristic threshold baseline detector
│   ├── evaluate.py           # Metrics, confusion matrices, and SHAP explainability
│   └── realtime.py           # Standalone real-time webcam detector loop
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   └── main.py      # FastAPI service (WebSocket landmark ingestion & alerts)
│   └── frontend/
│       └── index.html   # Web-based monitoring UI with disclaimer & live feedback
├── models/              # Serialized trained model pipelines (.joblib, git-ignored)
├── reports/             # Evaluation charts, confusion matrices, and metrics summaries
├── .env.example         # Sample environment configurations
├── .gitignore           # Git ignore rules for data, models, logs, and environments
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation and setup guide
```

---

## Virtual Environment Setup Guide

### 1. Prerequisites
- **Python 3.10** or **Python 3.11** installed.
- Ensure `pip` is updated to the latest version.

Verify your installed Python version:
```bash
python --version
# Should output Python 3.10.x or Python 3.11.x
```

### 2. Create the Virtual Environment

Open a terminal in the project root (`fallguard/`):

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*Note: If PowerShell displays an execution policy error, enable local script execution with:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Once the virtual environment is activated:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the provided example configuration file:
```bash
# Windows (PowerShell or CMD)
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Review and adjust settings in `.env` (such as `PORT=8000`, `PROBABILITY_THRESHOLD=0.7`, `LYING_SECONDS=2.0`).

---

## Run the App

A single command boots the FastAPI backend and simultaneously serves the frontend dashboard at `http://localhost:8000`.

### 1. Start the Server

Ensure your virtual environment is activated, then launch:

#### On Windows (PowerShell or Command Prompt):
```powershell
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Alternatively, run via the virtual environment Python executable:*
```powershell
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### On Linux / macOS:
```bash
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Application in Your Browser

Open your browser and navigate to:
- **Home & Architecture Overview:** [http://localhost:8000](http://localhost:8000) or [http://localhost:8000/index.html](http://localhost:8000/index.html)
- **Live Pose Detection Monitor:** [http://localhost:8000/monitor.html](http://localhost:8000/monitor.html)
- **Ethics & Privacy Breakdown:** [http://localhost:8000/about.html](http://localhost:8000/about.html)
- **REST Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- **Alerts Feed:** [http://localhost:8000/alerts](http://localhost:8000/alerts)

On the **Live Monitor** page, click **"Start Monitoring"** to grant webcam access. The interface connects to `ws://localhost:8000/ws`, displays a green **"Connected"** indicator, and begins streaming numerical landmark coordinates at 10 Hz with zero raw video stored or transmitted.

---


## Module Overview

| File | Purpose |
| :--- | :--- |
| `src/extract_landmarks.py` | Runs MediaPipe Pose on raw video files in `data/raw/` and outputs numerical landmark series to `data/landmarks/`. No raw video is kept. |
| `src/features.py` | Calculates torso angle, hip height, vertical speed, aspect ratio (normalized by torso length), and computes 2-second sliding window summary statistics. Shared across training and inference. |
| `src/baseline_rules.py` | Physics-based heuristic fall detection baseline for benchmarking ML models. |
| `src/train.py` | Trains Random Forest and XGBoost classifiers using `GroupKFold` by video ID. |
| `src/evaluate.py` | Evaluates trained models, generates ROC curves, confusion matrices, and SHAP importance plots. |
| `src/realtime.py` | Live webcam detection script applying two-stage confirmation on CPU. |
| `app/backend/main.py` | FastAPI application serving real-time fall detection events and WebSockets. |
| `app/frontend/index.html` | Browser dashboard displaying real-time skeleton overlay, alerts, and medical disclaimer. |
