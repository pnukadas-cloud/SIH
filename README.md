# 🛰️ MAITRI — Multimodal AI Assistant for Astronaut Psychological & Physical Well-Being

[![MAITRI CI & Telemetry Verification](https://github.com/pnukadas-cloud/SIH/actions/workflows/ci.yml/badge.svg)](https://github.com/pnukadas-cloud/SIH/actions/workflows/ci.yml)

> **Smart India Hackathon (SIH 2025)**  
> **Problem ID:** 25175 | **Category:** Software / Hard  
> **Target Agency:** Indian Space Research Organisation (ISRO) / Department of Space (DoS)  
> **Mission Ecosystem:** Bhartiya Antariksh Station (BAS) / Gaganyaan

---

## 🌌 Overview

Crew members aboard long-duration space stations face acute isolation, circadian disruption, high-consequence operational pressure, and microgravity physical discomforts. **MAITRI** is an **edge-deployable, standalone offline multimodal AI assistant** designed to continuously monitor astronaut psychological and physical well-being using audio-video telemetry, provide evidence-based psychological interventions, and report critical risk alerts to ground control.

---

## ✨ Key Features

1. **Multimodal Emotion Recognition (7 Standardized Classes)**:
   - **Facial Emotion Recognition (FER)**: Analyzes facial landmarks, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Facial Action Units (AU4, AU6, AU12, AU20, AU25, AU43).
   - **Speech Emotion Recognition (SER)**: Processes fundamental frequency (F0 pitch), RMS energy, spectral centroid, spectral rolloff, and vocal jitter/shimmer.
   - **Linguistic Sentiment Engine**: Analyzes verbal transcripts for cognitive overload, space isolation markers, and affective sentiment.

2. **Attention-Weighted Late Fusion with Cross-Modal Discordance**:
   - Integrates optical, acoustic, and linguistic channels with dynamic confidence weighting.
   - **Masked Distress Detection**: Detects when an astronaut smiles facially while vocal tension/prosody exhibits acute stress.

3. **Physical Distress & Fatigue Monitoring**:
   - **PERCLOS** (Percentage of Eye Closure time) for drowsiness and microsleep risk estimation.
   - Yawning frequency detection and blink rate anomaly tracking.
   - Facial grimacing / pain biomarker estimation (AU4 + AU7 + AU20).

4. **4-Tier Risk Escalation Framework**:
   - 🟢 **Level 0 (Score 0–30)**: Nominal baseline / Passive monitoring.
   - 🟡 **Level 1 (Score 31–50)**: Mild Concern / Proactive companion check-in.
   - 🟠 **Level 2 (Score 51–70)**: Moderate Risk / Active clinical intervention + Ground station queue.
   - 🔴 **Level 3 (Score 71–100)**: Critical Alarm / Emergency ground beacon + Crisis grounding protocol.

5. **Evidence-Based Psychological Interventions**:
   - **Box Breathing (Sama Vritti 4-4-4-4)** with real-time visual orbital pacer.
   - **5-4-3-2-1 Sensory Grounding** adapted for space station cabins.
   - **CBT Cognitive Reframing** for high-tempo mission pressure.
   - **Micro-Rest & Circadian Alignment Protocol** for fatigue mitigation.

6. **Offline AI Space Companion & TTS**:
   - Offline conversational intelligence tailored for astronaut well-being.
   - Offline Text-to-Speech (TTS) for space cabin audio responses.

7. **ISRO Ground Station Alert Dispatcher**:
   - Automatically builds standardized telemetry packets queued for ISTRAC / IDRSS data relay passes.

---

## 🏗️ Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │             INPUT TELEMETRY LAYER            │
                      │   📹 Optical Video Feed   🎤 Acoustic Mic    │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │            PREPROCESSING PIPELINE            │
                      │  • Face Detection & Action Units (EAR/MAR)   │
                      │  • Voice Activity & Pitch / Jitter Profiling │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │          MULTIMODAL ANALYSIS ENGINE          │
                      │   FER (Vision) ── SER (Audio) ── NLP (Text)  │
                      │             Physical Distress Vitals         │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │    ATTENTION-WEIGHTED MULTIMODAL FUSION      │
                      │  E_fused = α·Face + β·Voice + γ·Linguistic   │
                      │     + Masked Distress Discordance Check      │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │         DECISION & STATE TRACKER             │
                      │  • Rolling Temporal State Window (5 mins)    │
                      │  • 4-Tier Risk Scorer (Levels 0, 1, 2, 3)    │
                      │  • Context Memory (SQLite Flight Recorder)   │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │              RESPONSE ENGINE                 │
                      │  • Offline Conversational AI Companion       │
                      │  • Clinical Interventions (Breathing/CBT)    │
                      │  • Offline Text-to-Speech (TTS)              │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │         OUTPUT & GROUND STATION DISPATCH     │
                      │  🖥️ BAS Spacecraft HUD   📡 ISRO Ground Log  │
                      └──────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Running Locally

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Launch MAITRI
```bash
python run_maitri.py
```
*Your browser will automatically open to `http://127.0.0.1:8000` with the Spacecraft HUD.*

---

## 🎮 Interactive Live Demo & Simulation Modes

MAITRI includes built-in flight simulation scenarios accessible directly from the top toolbar:
1. **🟢 1. Nominal Orbit**: Demonstrates baseline calm/nominal state with green vitals.
2. **🟠 2. Docking Stress (Level 2)**: Triggers acute procedural stress with brow furrowing and high vocal tension.
3. **🔵 3. Space Isolation**: Simulates psychological loneliness and detachment with compassionate AI companion dialogue.
4. **🔴 4. Severe Fatigue & Microsleep (Level 3)**: Demonstrates PERCLOS > 35%, yawn spikes, and triggers automatic emergency rest guidance + ground alert.
5. **⚠️ 5. Masked Distress**: Demonstrates cross-modal discordance detection (astronaut smiling while voice pitch exhibits acute tension).

---

## 📦 Project Directory Structure

```
SIH/
├── maitri/
│   ├── config.py                 # System configurations and thresholds
│   ├── pipeline.py               # Master Pipeline Orchestrator
│   ├── preprocessing/
│   │   ├── vision_processor.py   # Face, eye, mouth & action unit extraction
│   │   └── audio_processor.py    # Pitch, jitter, spectral centroid & VAD
│   ├── analysis/
│   │   ├── fer_engine.py         # 7-Class Facial Emotion Recognition
│   │   ├── ser_engine.py         # Speech Emotion Recognition
│   │   ├── text_sentiment_engine.py # Linguistic & Cognitive Overload analyzer
│   │   ├── physical_distress_engine.py # Fatigue, PERCLOS & Pain detection
│   │   └── multimodal_fusion.py  # Attention-Weighted Late Fusion
│   ├── decision/
│   │   ├── state_tracker.py      # Rolling temporal state tracker
│   │   ├── risk_scorer.py        # 4-Tier Risk Escalation Engine
│   │   └── context_memory.py     # SQLite persistence & flight recorder
│   ├── response/
│   │   ├── interventions.py      # Evidence-based psychological protocols
│   │   ├── conversational_agent.py # Empathetic AI Companion
│   │   └── tts_engine.py         # Offline Text-to-Speech
│   ├── telemetry/
│   │   ├── ground_station.py     # ISRO Ground Station alert dispatcher
│   │   └── mission_logger.py     # Session event logger
│   ├── data/
│   │   ├── interventions_db.json # Clinical intervention protocols
│   │   └── astronaut_baselines.json # Gaganyaan/BAS crew baselines
│   └── web/
│       ├── static/               # CSS & JavaScript for Spacecraft HUD
│       └── templates/            # HTML Template for Mission Control
├── server.py                     # FastAPI REST & WebSocket Server
├── run_maitri.py                 # One-click launcher
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🏆 SIH Evaluation Alignment

| Requirement | Implementation in MAITRI |
|---|---|
| **Audio-Visual Input Processing** | Real-time optical face tracking + acoustic pitch/jitter feature extraction. |
| **Multimodal Emotion Detection** | Attention-weighted late fusion across FER, SER, and Linguistic sentiment. |
| **Physical Distress Detection** | PERCLOS eye closure, yawn frequency, blink rate anomalies, and pain grimace index. |
| **Adaptive Supportive Dialogues** | Context-aware empathetic AI companion with space station situation awareness. |
| **Evidence-Based Interventions** | Box breathing pacer, 5-4-3-2-1 grounding, CBT reframing, and micro-rest protocols. |
| **Ground Station Reporting** | Standardized JSON telemetry alert packet generation with queue management. |
| **Offline Standalone System** | 100% offline edge execution with zero cloud dependency. |
