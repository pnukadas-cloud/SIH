# 🚀 MAITRI

> Multimodal AI Assistant for Psychological & Physical
> Well-Being of Astronauts

[SIH 2025] [Problem ID: 25175] [ISRO / Department of Space]

## 🌌 About

Short 1–2 paragraph explanation of the problem and MAITRI.

## ✨ Key Features

- 🎥 Facial emotion recognition
- 🎤 Speech emotion recognition
- 📝 Text sentiment analysis
- 🔀 Multimodal emotion fusion
- 🩺 Physical distress detection
- 🧠 Temporal emotion & risk tracking
- 💬 Local LLM-based conversations
- 🧘 Evidence-based interventions
- 🔊 Offline text-to-speech
- 🚨 Ground station alerts
- 🔒 Fully offline / edge computing

## 🏗️ Architecture

[Simple architecture diagram]

Camera + Mic
     ↓
Preprocessing
     ↓
FER + SER + Sentiment
     ↓
Multimodal Fusion
     ↓
Emotion + Risk Engine
     ↓
Local LLM + Intervention
     ↓
TTS → Astronaut
     ↓
Ground Station Alerts

## 🛠️ Tech Stack

Python | OpenCV | MediaPipe | Whisper
PyTorch | DistilBERT | Phi-3 Mini
llama.cpp | Piper TTS | FastAPI | SQLite

## 🚨 Risk Levels

| Level | Score | Response |
|---|---|---|
| 🟢 0 | 0–30 | Passive monitoring |
| 🟡 1 | 31–50 | Proactive check-in |
| 🟠 2 | 51–70 | Intervention + ground alert |
| 🔴 3 | 71–100 | Crisis protocol |


## ⚙️ Installation

```bash
git clone ...
cd MAITRI
pip install -r requirements.txt
