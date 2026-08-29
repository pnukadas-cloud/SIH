# DevOps Domain

## Deployment Overview
This domain provides containerization, orchestration, environment configuration, and test automation for MAITRI.

## Folder Structure
- **`docker/`**:
  - `Dockerfile`: Multi-stage Python 3.11-slim container with OpenCV, ReportLab, and libsndfile.
  - `docker-compose.yml`: Standalone container setup with volume mounts and automated health checks.
- **`configuration/`**:
  - `.env.example`: Secure environment configuration template.
- **`scripts/`**:
  - `run_tests.py`: Complete automated test runner checking API status, RBAC enforcement, PDF/JSON/JPG generation, and conversational companion AI.

## Quickstart Commands

### 1. Local Run
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

### 2. Docker Compose
```bash
docker-compose -f DevOps/docker/docker-compose.yml up --build -d
```

### 3. Run Automated Tests
```bash
python DevOps/scripts/run_tests.py
```
