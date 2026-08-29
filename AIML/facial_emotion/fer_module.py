"""
AIML — Robust Facial Emotion & State Recognition Module
Phase 2 Implementation:
Camera -> Face Detection -> Face Quality Check -> Facial Features / Action Units ->
Personal Baseline Comparison -> Temporal Smoothing & Hysteresis -> Stable Video State.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional

class FacialEmotionModule:
    def __init__(self):
        self.classes = ['neutral', 'happy', 'stressed', 'fatigued', 'anxious', 'sad', 'frustrated']
        self.state_labels = {
            'neutral': 'relaxed',
            'happy': 'expressive_positive',
            'stressed': 'strained',
            'fatigued': 'fatigued',
            'anxious': 'attentive_vigilant',
            'sad': 'withdrawn',
            'frustrated': 'strained'
        }

        self.active_astronaut_id: str = "CREW-BAS-01"
        self.personal_baselines: Dict[str, Dict[str, float]] = {}

        # Rolling temporal tracking buffers (60 second window for rates)
        self.blink_timestamps: List[float] = []
        self.yawn_timestamps: List[float] = []
        self.ear_history: List[Tuple[float, float]] = []
        self.mar_history: List[Tuple[float, float]] = []
        self.last_face_box: Optional[Tuple[int, int, int, int]] = None

        # Temporal Smoothing & State Hysteresis (Phase 2)
        self.ema_probabilities: Dict[str, float] = {c: 1.0 / len(self.classes) for c in self.classes}
        self.ema_probabilities["neutral"] = 0.70
        tot = sum(self.ema_probabilities.values())
        self.ema_probabilities = {k: v / tot for k, v in self.ema_probabilities.items()}

        self.recent_state_history: List[str] = ["neutral"] * 5
        self.stable_dominant_emotion: str = "neutral"
        self.stable_facial_state: str = "relaxed"
        self.stable_confidence: float = 0.85
        self.candidate_state: str = "neutral"
        self.candidate_count: int = 0
        self.required_persistence_frames: int = 3  # ~0.75s to 1.0s persistence

    def set_active_astronaut(self, astronaut_id: str, baseline: Optional[Dict[str, Any]] = None):
        """Associate pipeline with specific astronaut and load their personal baseline."""
        if self.active_astronaut_id != astronaut_id:
            self.active_astronaut_id = astronaut_id
            self.blink_timestamps.clear()
            self.yawn_timestamps.clear()
            self.ear_history.clear()
            self.mar_history.clear()
            self.recent_state_history = ["neutral"] * 5
            self.stable_dominant_emotion = "neutral"
            self.stable_facial_state = "relaxed"
            self.candidate_state = "neutral"
            self.candidate_count = 0
            self.ema_probabilities = {c: (0.70 if c == 'neutral' else 0.05) for c in self.classes}
            tot = sum(self.ema_probabilities.values())
            self.ema_probabilities = {k: v / tot for k, v in self.ema_probabilities.items()}

        if baseline:
            self.personal_baselines[astronaut_id] = {
                "resting_ear": float(baseline.get("resting_ear", 0.32)),
                "resting_mar": float(baseline.get("resting_mar", 0.18)),
                "blink_rate_bpm": float(baseline.get("blink_rate_bpm", 16.0)),
                "resting_au04": float(baseline.get("resting_au04", 0.10))
            }

    def _get_baseline(self, astronaut_id: str) -> Dict[str, float]:
        """Return personal baseline or default normal values."""
        return self.personal_baselines.get(astronaut_id, {
            "resting_ear": 0.32,
            "resting_mar": 0.18,
            "blink_rate_bpm": 16.0,
            "resting_au04": 0.10
        })

    def assess_face_quality(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        faces: List[Tuple[int, int, int, int, float, bool]]
    ) -> Dict[str, Any]:
        """
        Step 3: Rigorous Face Quality Evaluation.
        Detects blur, poor lighting, extreme head pose, small face area, and occlusions.
        """
        h, w = frame.shape[:2]
        mean_brightness = float(np.mean(gray)) if gray.size > 0 else 0.0
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 0 else 0.0

        if not faces:
            status = "POOR_LIGHTING" if (5.0 <= mean_brightness < 28.0) else "NO_FACE"
            msg = "Low cabin illumination; optical features degraded." if status == "POOR_LIGHTING" else "No astronaut face detected in camera viewport."
            return {
                "valid": False,
                "quality_score": 0.0,
                "status": status,
                "brightness": round(mean_brightness, 1),
                "sharpness": round(laplacian_var, 1),
                "face_area_ratio": 0.0,
                "message": msg
            }

        fx, fy, fw, fh, area, is_fallback = faces[0]
        face_area = fw * fh
        frame_area = w * h
        face_area_ratio = float(face_area) / max(1.0, float(frame_area))
        aspect_ratio = float(fh) / max(1.0, float(fw))

        # Check face region brightness specifically
        face_roi = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        face_brightness = float(np.mean(face_roi)) if face_roi.size > 0 else mean_brightness

        status = "OPTIMAL"
        msg = "Nominal optical clarity and illumination."
        valid = True
        quality_score = 1.0

        # 1. Lighting check (mean < 28 or face < 30 is poor; > 230 is overexposed)
        if mean_brightness < 28.0 or face_brightness < 30.0:
            status = "POOR_LIGHTING"
            msg = "Low cabin illumination; optical features degraded."
            valid = False
            quality_score *= 0.35
        elif mean_brightness > 230 or face_brightness > 235:
            status = "OVEREXPOSED"
            msg = "Severe optical glare or overexposure detected."
            valid = False
            quality_score *= 0.40

        # 2. Extreme head pose / aspect ratio
        if aspect_ratio < 0.88 or aspect_ratio > 2.15:
            status = "EXTREME_POSE" if status == "OPTIMAL" else status
            msg = "Extreme head pose or profile angle detected; landmarks asymmetric."
            valid = False
            quality_score *= 0.35

        # 3. Partial occlusion / fragmented contour
        contour_density = float(area) / max(1.0, float(face_area))
        if contour_density < 0.65:
            status = "OCCLUSION" if status == "OPTIMAL" else status
            msg = "Partial facial occlusion detected (hand or visor obstruction)."
            valid = False
            quality_score *= 0.40

        # 4. Sharpness / Blur check
        if laplacian_var < 15.0:
            status = "BLURRED" if status == "OPTIMAL" else status
            msg = "Motion blur or optical defocus detected."
            valid = False
            quality_score *= 0.45

        # 5. Face size check
        if face_area_ratio < 0.030:
            status = "FACE_TOO_SMALL" if status == "OPTIMAL" else status
            msg = "Astronaut is too far from camera for accurate landmark extraction."
            valid = False
            quality_score *= 0.40

        return {
            "valid": valid,
            "quality_score": round(float(np.clip(quality_score, 0.0, 1.0)), 3),
            "status": status,
            "brightness": round(mean_brightness, 1),
            "sharpness": round(laplacian_var, 1),
            "face_area_ratio": round(face_area_ratio, 3),
            "message": msg
        }

    def detect_face_multispectral(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int, float, bool]]:
        """
        Step 2: Multi-Spectral Skin Chrominance Face Localization (YCrCb + HSV).
        Returns list of (x, y, w, h, contour_area, is_fallback).
        """
        h, w = frame.shape[:2]
        min_face_area = int(w * h * 0.025)
        max_face_area = int(w * h * 0.90)

        # 1. YCrCb Skin Filter
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))

        # 2. HSV Skin Filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8), np.array([28, 255, 255], dtype=np.uint8))

        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Morphological operations to bridge facial contours
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_face_area <= area <= max_face_area:
                x, y, fw, fh = cv2.boundingRect(c)
                candidates.append((x, y, fw, fh, area, False))

        if candidates:
            # Sort by area descending
            candidates.sort(key=lambda item: item[4], reverse=True)
            self.last_face_box = (candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3])
            return candidates

        # Center Prior Fallback only if sufficient ambient light exists and no contour found
        mean_val = float(np.mean(gray)) if gray.size > 0 else 0.0
        if mean_val > 45.0:
            cw = int(w * 0.42)
            ch = int(h * 0.58)
            cx = int((w - cw) / 2)
            cy = int((h - ch) / 2.3)
            self.last_face_box = (cx, cy, cw, ch)
            return [(cx, cy, cw, ch, cw * ch * 0.75, True)]

        return []

    def extract_features(
        self,
        frame: Optional[np.ndarray],
        astronaut_id: Optional[str] = None,
        astronaut_baseline: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete End-to-End Pipeline:
        Camera -> Face Detection -> Quality Check -> Facial Features / Action Units ->
        Personal Baseline Comparison -> Temporal Smoothing & Hysteresis -> Stable Output.
        """
        now = time.time()
        if astronaut_id:
            self.set_active_astronaut(astronaut_id, astronaut_baseline)

        baseline = self._get_baseline(self.active_astronaut_id)

        # 1. Validate Frame
        if frame is None or frame.size == 0:
            return self._empty_features(now, "No video stream received", baseline)

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 2. Detect Faces
        faces = self.detect_face_multispectral(frame, gray)

        # 3. Assess Face Quality
        face_quality = self.assess_face_quality(frame, gray, faces)

        # If quality is invalid, do NOT generate wild emotional changes; decay gracefully
        if not face_quality["valid"]:
            return self._quality_degraded_features(now, face_quality, baseline)

        fx, fy, fw, fh = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
        face_roi = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        if face_roi.size == 0:
            return self._empty_features(now, "Empty face region", baseline)

        rh, rw = face_roi.shape[:2]

        # -------------------------------------------------------------
        # 4. Extract Facial Features & Action Units
        # -------------------------------------------------------------
        # Eye Aspect Ratio (EAR)
        eye_roi = face_roi[int(rh * 0.20):int(rh * 0.48), int(rw * 0.12):int(rw * 0.88)]
        ear = baseline["resting_ear"]
        if eye_roi.size > 0:
            _, eye_thresh = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            eye_vert = np.sum(eye_thresh, axis=1) / max(1.0, eye_roi.shape[1])
            active_h = np.count_nonzero(eye_vert > 20)
            ear_est = active_h / max(1.0, float(eye_roi.shape[0]))
            ear = float(np.clip(0.12 + (ear_est * 0.35), 0.12, 0.44))

        self.ear_history.append((now, ear))
        if len(self.ear_history) >= 2 and ear < 0.21 and self.ear_history[-2][1] >= 0.21:
            self.blink_timestamps.append(now)

        # Mouth Aspect Ratio (MAR) & Smile (AU12)
        mouth_roi = face_roi[int(rh * 0.62):int(rh * 0.95), int(rw * 0.15):int(rw * 0.85)]
        mar = baseline["resting_mar"]
        smile_intensity = 0.0
        if mouth_roi.size > 0:
            _, mouth_thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mouth_vert = np.sum(mouth_thresh, axis=1) / max(1.0, mouth_roi.shape[1])
            mouth_h = np.count_nonzero(mouth_vert > 25)
            mar = float(np.clip(mouth_h / max(1.0, float(mouth_roi.shape[0])), 0.10, 0.85))

            # Mouth active horizontal span relative to ROI
            mouth_horiz = np.sum(mouth_thresh, axis=0) / max(1.0, mouth_roi.shape[0])
            active_width_ratio = float(np.count_nonzero(mouth_horiz > 10)) / max(1.0, float(mouth_roi.shape[1]))
            # Smile expands mouth width significantly (>0.80) with moderate openness (<0.75)
            if active_width_ratio > 0.80 and mar < 0.75:
                smile_intensity = float(np.clip((active_width_ratio - 0.78) * 4.5, 0.0, 1.0))

        self.mar_history.append((now, mar))
        if mar > 0.52 and (not self.yawn_timestamps or (now - self.yawn_timestamps[-1] > 3.0)):
            self.yawn_timestamps.append(now)

        # Forehead Strain / Brow Furrow (AU04) on Glabella region
        # Focused specifically between eyebrows to avoid boundary hairline artifacts
        glabella_roi = face_roi[int(rh * 0.20):int(rh * 0.38), int(rw * 0.35):int(rw * 0.65)]
        brow_furrow = baseline["resting_au04"]
        if glabella_roi.size > 0:
            sobely = cv2.Sobel(glabella_roi, cv2.CV_64F, 0, 1, ksize=3)
            sobely_var = float(np.var(sobely))
            brow_furrow = float(np.clip((sobely_var - 8000.0) / 25000.0, 0.05, 0.95))

        # Purge rolling buffers older than 60s
        cutoff = now - 60.0
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff]
        self.yawn_timestamps = [t for t in self.yawn_timestamps if t >= cutoff]
        self.ear_history = [(t, v) for t, v in self.ear_history if t >= cutoff]
        self.mar_history = [(t, v) for t, v in self.mar_history if t >= cutoff]

        perclos = float(sum(1 for _, e in self.ear_history if e < 0.21) / max(1, len(self.ear_history)))
        blinks_per_min = float(len(self.blink_timestamps))
        yawns_per_min = len(self.yawn_timestamps)

        # -------------------------------------------------------------
        # 5. Personal Baseline Comparison
        # -------------------------------------------------------------
        base_ear = baseline["resting_ear"]
        base_mar = baseline["resting_mar"]
        base_blinks = baseline["blink_rate_bpm"]
        base_au04 = baseline["resting_au04"]

        delta_ear = ear - base_ear
        delta_mar = mar - base_mar
        delta_blinks = blinks_per_min - base_blinks
        delta_au04 = brow_furrow - base_au04

        # Adaptive baseline gentle update during high-quality nominal periods
        if face_quality["quality_score"] > 0.85 and brow_furrow < 0.25 and perclos < 0.05:
            baseline["resting_ear"] = round(base_ear * 0.99 + ear * 0.01, 3)
            baseline["resting_mar"] = round(base_mar * 0.99 + mar * 0.01, 3)
            self.personal_baselines[self.active_astronaut_id] = baseline

        # -------------------------------------------------------------
        # 6. Frame-Level Observation Probabilities
        # -------------------------------------------------------------
        frame_probs = {c: 0.04 for c in self.classes}

        # Stress indicator: brow furrow elevation + excessive blink or mouth tension
        stress_indicator = float(np.clip(max(0.0, delta_au04 * 1.5) + (0.3 if delta_blinks > 8 else 0.0), 0.0, 1.0))
        # Fatigue indicator: eye droop (negative delta EAR) + PERCLOS + yawns
        fatigue_indicator = float(np.clip((perclos * 2.5) + max(0.0, -delta_ear * 2.0) + (yawns_per_min * 0.2), 0.0, 1.0))

        if smile_intensity > 0.35 and stress_indicator < 0.3:
            frame_probs["happy"] = 0.55 + (smile_intensity * 0.35)
            frame_probs["neutral"] = 0.15
        elif stress_indicator > 0.35:
            frame_probs["stressed"] = 0.50 + (stress_indicator * 0.40)
            frame_probs["frustrated"] = 0.20
            frame_probs["neutral"] = 0.10
        elif fatigue_indicator > 0.35:
            frame_probs["fatigued"] = 0.50 + (fatigue_indicator * 0.40)
            frame_probs["neutral"] = 0.15
        elif delta_blinks > 10.0:
            frame_probs["anxious"] = 0.45
            frame_probs["neutral"] = 0.25
        else:
            frame_probs["neutral"] = 0.75
            frame_probs["happy"] = 0.10
            frame_probs["fatigued"] = 0.07

        # Normalize frame probabilities
        frame_tot = sum(frame_probs.values())
        norm_frame_probs = {k: v / frame_tot for k, v in frame_probs.items()}

        # -------------------------------------------------------------
        # 7. Temporal Smoothing & State Hysteresis (Phase 2 Core)
        # -------------------------------------------------------------
        # Exponential Moving Average (alpha = 0.25, smoothing window ~2.5s)
        alpha = 0.25
        for c in self.classes:
            self.ema_probabilities[c] = (alpha * norm_frame_probs[c]) + ((1.0 - alpha) * self.ema_probabilities[c])

        ema_tot = sum(self.ema_probabilities.values())
        smoothed_probs = {k: round(v / ema_tot, 4) for k, v in self.ema_probabilities.items()}

        # Leading smoothed candidate
        leading_candidate = max(smoothed_probs.items(), key=lambda item: item[1])[0]

        # State Hysteresis: Require consecutive candidate persistence to avoid frame-flipping
        if leading_candidate == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = leading_candidate
            self.candidate_count = 1

        # Only transition stable state if candidate has persisted for N frames
        if self.candidate_count >= self.required_persistence_frames and smoothed_probs[leading_candidate] > 0.35:
            self.stable_dominant_emotion = leading_candidate
            self.stable_facial_state = self.state_labels.get(leading_candidate, "relaxed")

        self.recent_state_history.append(self.stable_dominant_emotion)
        if len(self.recent_state_history) > 10:
            self.recent_state_history.pop(0)

        # Temporal stability confidence
        consistency = float(self.recent_state_history.count(self.stable_dominant_emotion)) / len(self.recent_state_history)
        final_confidence = float(np.clip(smoothed_probs[self.stable_dominant_emotion] * 0.6 + consistency * 0.3 + face_quality["quality_score"] * 0.1, 0.45, 0.96))

        # Valence & Arousal
        valence = (smoothed_probs.get("happy", 0.0) * 0.9) - (smoothed_probs.get("stressed", 0.0) * 0.6) - (smoothed_probs.get("fatigued", 0.0) * 0.4) - (smoothed_probs.get("frustrated", 0.0) * 0.7)
        arousal = (smoothed_probs.get("stressed", 0.0) * 0.8) + (smoothed_probs.get("anxious", 0.0) * 0.7) + (smoothed_probs.get("happy", 0.0) * 0.5) - (smoothed_probs.get("fatigued", 0.0) * 0.6)

        action_units = {
            "AU04_brow_furrow": round(brow_furrow, 3),
            "AU06_cheek_raiser": round(smile_intensity * 0.75, 3),
            "AU12_lip_corner_puller": round(smile_intensity, 3),
            "AU20_lip_stretcher": round(float(np.clip(mar * 0.8, 0.0, 1.0)), 3),
            "AU25_lips_part": round(float(1.0 if mar > 0.35 else mar / 0.35), 3),
            "AU43_eye_closure": round(float(1.0 if ear < 0.21 else 0.0), 3)
        }

        return {
            "astronaut_id": self.active_astronaut_id,
            "facial_state": self.stable_facial_state,
            "dominant_emotion": self.stable_dominant_emotion,
            "stress_indicator": round(stress_indicator, 3),
            "fatigue_indicator": round(fatigue_indicator, 3),
            "facial_indicators": {
                "eye_aspect_ratio": round(ear, 3),
                "mouth_aspect_ratio": round(mar, 3),
                "brow_tension_au04": round(brow_furrow, 3),
                "smile_intensity_au12": round(smile_intensity, 3),
                "blink_rate_bpm": round(blinks_per_min, 1),
                "perclos": round(perclos, 4),
                "yawns_per_min": yawns_per_min
            },
            "confidence": round(final_confidence, 3),
            "face_quality": face_quality,
            "baseline_comparison": {
                "baseline_ear": round(base_ear, 3),
                "ear_deviation": round(delta_ear, 3),
                "baseline_mar": round(base_mar, 3),
                "mar_deviation": round(delta_mar, 3),
                "baseline_blinks": round(base_blinks, 1),
                "blink_deviation": round(delta_blinks, 1)
            },
            "face_detected": True,
            "face_count": len(faces),
            "multiple_faces": len(faces) > 1,
            "face_bounding_box": {"x": fx, "y": fy, "w": fw, "h": fh},
            "lighting": face_quality,
            "eye_aspect_ratio": round(ear, 3),
            "mouth_aspect_ratio": round(mar, 3),
            "blinks_per_min": round(blinks_per_min, 1),
            "yawns_per_min": yawns_per_min,
            "perclos": round(perclos, 4),
            "action_units": action_units,
            "probabilities": smoothed_probs,
            "valence": round(float(np.clip(valence, -1.0, 1.0)), 3),
            "arousal": round(float(np.clip(arousal, 0.0, 1.0)), 3),
            "modality_active": True,
            "timestamp": now
        }

    def _quality_degraded_features(self, timestamp: float, face_quality: Dict[str, Any], baseline: Dict[str, float]) -> Dict[str, Any]:
        """Gracefully hold previous stable state with lowered confidence when optical quality is degraded."""
        return {
            "astronaut_id": self.active_astronaut_id,
            "facial_state": self.stable_facial_state,
            "dominant_emotion": self.stable_dominant_emotion,
            "stress_indicator": 0.0,
            "fatigue_indicator": 0.0,
            "facial_indicators": {
                "eye_aspect_ratio": baseline["resting_ear"],
                "mouth_aspect_ratio": baseline["resting_mar"],
                "brow_tension_au04": baseline["resting_au04"],
                "smile_intensity_au12": 0.0,
                "blink_rate_bpm": baseline["blink_rate_bpm"],
                "perclos": 0.0,
                "yawns_per_min": 0
            },
            "confidence": 0.30,
            "face_quality": face_quality,
            "baseline_comparison": {
                "baseline_ear": baseline["resting_ear"],
                "ear_deviation": 0.0,
                "baseline_mar": baseline["resting_mar"],
                "mar_deviation": 0.0,
                "baseline_blinks": baseline["blink_rate_bpm"],
                "blink_deviation": 0.0
            },
            "face_detected": face_quality.get("status") in ["OPTIMAL", "OCCLUSION", "BLURRED", "FACE_TOO_SMALL"],
            "face_count": 1 if face_quality.get("status") in ["OPTIMAL", "OCCLUSION", "BLURRED", "FACE_TOO_SMALL"] else 0,
            "multiple_faces": False,
            "face_bounding_box": self.last_face_box,
            "lighting": face_quality,
            "eye_aspect_ratio": baseline["resting_ear"],
            "mouth_aspect_ratio": baseline["resting_mar"],
            "blinks_per_min": baseline["blink_rate_bpm"],
            "yawns_per_min": 0,
            "perclos": 0.0,
            "action_units": {au: 0.0 for au in ["AU04_brow_furrow", "AU06_cheek_raiser", "AU12_lip_corner_puller", "AU20_lip_stretcher", "AU25_lips_part", "AU43_eye_closure"]},
            "probabilities": self.ema_probabilities,
            "valence": 0.0,
            "arousal": 0.15,
            "modality_active": False,
            "timestamp": timestamp
        }

    def _empty_features(self, timestamp: float, reason: str, baseline: Dict[str, float]) -> Dict[str, Any]:
        return self._quality_degraded_features(
            timestamp=timestamp,
            face_quality={
                "valid": False,
                "quality_score": 0.0,
                "status": "NO_FACE",
                "brightness": 0.0,
                "sharpness": 0.0,
                "face_area_ratio": 0.0,
                "message": reason
            },
            baseline=baseline
        )
