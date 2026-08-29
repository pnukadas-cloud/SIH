# AIML Domain — Multimodal AI Pipeline

## Architecture Overview
The `AIML` domain is responsible for end-to-end multimodal perception, emotion classification, autonomic feature fusion, and empathetic dialogue generation.

```
Camera Frame (JPEG/RGB)  ───> facial_emotion/  (FACS Action Units, EAR, MAR, Lighting)
                                     │
Microphone Audio (PCM)    ───> speech_acoustics/ (Autocorrelation F0 Pitch, Jitter, RMS)
                                     │
Spoken Transcript (Text)  ───> NLP Sentiment
                                     │
                                     ▼
                          emotional_valence/ (Late-Attention Dynamic Fusion)
                                     │
                                     ├───> Val (-1 to +1), Arousal (0 to 1), Discordance
                                     │
                                     ▼
                              wellbeing/ (Well-Being Index 0–100 & 4-Tier Status)
                                     │
                                     ▼
                                maitri/ (Companion AI: Gemini 1.5 Flash + Offline KB)
```

## Module Specifications

### 1. `facial_emotion/fer_module.py`
- **Inputs**: Video frame (`np.ndarray`).
- **Environmental Checks**: Evaluates mean pixel intensity (<35 indicates Low Illumination) and Laplacian variance (<20 indicates Blur). Checks for multiple faces and tracks primary ROI.
- **Biomarkers**:
  - `Eye Aspect Ratio (EAR)`: Tracks microsleep and blink rates.
  - `Mouth Aspect Ratio (MAR)`: Tracks yawning (>0.52) and smile intensity.
  - `PERCLOS`: Percentage of eye closure over a rolling 60-second window.
  - `FACS Action Units`: AU04 (brow furrow), AU06 (cheek raiser), AU12 (lip corner puller), AU20 (lip stretcher), AU43 (eye closure).
- **Output**: 7-class emotion probabilities (`neutral`, `happy`, `stressed`, `fatigued`, `anxious`, `sad`, `frustrated`), facial valence, and quality-adjusted confidence.

### 2. `speech_acoustics/ser_module.py`
- **Inputs**: Raw PCM audio signal.
- **Speech Activity (VAD)**: Compares RMS energy against ambient noise floor (`0.015`). If silent, flags `is_speech_active: False`.
- **Acoustic Metrics**:
  - `Autocorrelation F0 Pitch`: Fundamental vocal frequency between 60 Hz and 450 Hz.
  - `Vocal Tension`: Normalized shift above baseline pitch combined with high-frequency energy.
  - `Jitter`: Micro-cycle frequency perturbations.
  - `Spectral Centroid`: Frequency center of mass.
- **Output**: Acoustic valence, arousal, and speech emotion probabilities.

### 3. `emotional_valence/fusion_module.py`
- **Attention Fusion**: Weights active modalities dynamically by confidence:
  $$\alpha \cdot P_{face} + \beta \cdot P_{voice} + \gamma \cdot P_{text}$$
- **Cross-Modal Discordance**: Detects masked stress (e.g. smiling face with tense voice) and verbal minimization.

### 4. `wellbeing/wellbeing_evaluator.py`
- **Formula**:
  $$\text{Score} = \text{Negative Affect (35\%)} + \text{Ocular Fatigue (30\%)} + \text{Autonomic Tension (25\%)} + \text{Discordance (10\%)}$$
- **ISRO Tiers**:
  - Level 0: Nominal / Rested (0–30)
  - Level 1: Mild Cognitive Load (31–50)
  - Level 2: Moderate Distress / High Fatigue (51–70)
  - Level 3: Acute Critical Distress (71–100)

### 5. `maitri/companion_ai.py`
- **Online**: Integrates Google Gemini 1.5 Flash via REST API when `GEMINI_API_KEY` is provided.
- **Offline**: Contextual Psychological Support Engine with multi-turn memory and clinical intervention recommendations.
