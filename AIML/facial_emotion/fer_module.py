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

        # Optical Detectors (Haar Cascade Classifiers)
        self.face_cascade = None
        self.face_cascade_alt = None
        self.eye_cascade = None
        self.smile_cascade = None
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            except Exception as e:
                print(f"[FER] Cascade loading warning: {e}")

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
        self.required_persistence_frames: int = 2  # Fast ~0.5s transition for real-time responsiveness

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
                "resting_ear": float(baseline.get("resting_ear", 0.28)),
                "resting_mar": float(baseline.get("resting_mar", 0.18)),
                "blink_rate_bpm": float(baseline.get("blink_rate_bpm", 16.0)),
                "resting_au04": float(baseline.get("resting_au04", 0.10))
            }

    def _get_baseline(self, astronaut_id: str) -> Dict[str, float]:
        """Return personal baseline or default normal values."""
        return self.personal_baselines.get(astronaut_id, {
            "resting_ear": 0.28,
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
        Step 3: Face Quality Evaluation.
        """
        h, w = frame.shape[:2]
        mean_brightness = float(np.mean(gray)) if gray.size > 0 else 0.0
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 0 else 0.0

        if not faces:
            status = "POOR_LIGHTING" if (5.0 <= mean_brightness < 25.0) else "NO_FACE"
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

        face_roi = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        face_brightness = float(np.mean(face_roi)) if face_roi.size > 0 else mean_brightness

        status = "OPTIMAL"
        msg = "Nominal optical clarity and illumination."
        valid = True
        quality_score = 1.0

        if mean_brightness < 20.0 or face_brightness < 22.0:
            status = "POOR_LIGHTING"
            msg = "Low cabin illumination; optical features degraded."
            valid = False
            quality_score *= 0.35
        elif mean_brightness > 245 or face_brightness > 248:
            status = "OVEREXPOSED"
            msg = "Severe optical glare or overexposure detected."
            valid = False
            quality_score *= 0.40

        if aspect_ratio < 0.70 or aspect_ratio > 2.30:
            status = "EXTREME_POSE" if status == "OPTIMAL" else status
            msg = "Extreme head pose or profile angle detected."
            valid = False
            quality_score *= 0.45

        if face_area_ratio < 0.020:
            status = "FACE_TOO_SMALL" if status == "OPTIMAL" else status
            msg = "Astronaut is too far from camera for accurate feature extraction."
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
        Robust Face Localization:
        Primary: Multi-scale Haar cascade classifier.
        Secondary: Alt2 Haar cascade classifier.
        Tertiary: Skin chrominance contours.
        Returns list of (x, y, w, h, contour_area, is_fallback).
        """
        h, w = frame.shape[:2]
        min_w = int(w * 0.12)
        min_h = int(h * 0.12)

        # 1. Primary Haar frontalface
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(min_w, min_h)
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                candidates = [(int(x), int(y), int(fw), int(fh), float(fw * fh), False) for (x, y, fw, fh) in faces]
                self.last_face_box = (candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3])
                return candidates

        # 2. Secondary Haar frontalface alt2
        if self.face_cascade_alt is not None and not self.face_cascade_alt.empty():
            faces = self.face_cascade_alt.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(min_w, min_h)
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                candidates = [(int(x), int(y), int(fw), int(fh), float(fw * fh), False) for (x, y, fw, fh) in faces]
                self.last_face_box = (candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3])
                return candidates

        # 3. Skin Chrominance Fallback
        min_face_area = int(w * h * 0.04)
        max_face_area = int(w * h * 0.85)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 20, 40], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8))
        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

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
                aspect = fh / max(1.0, float(fw))
                if 0.75 <= aspect <= 1.85:
                    candidates.append((x, y, fw, fh, area, False))

        if candidates:
            candidates.sort(key=lambda item: item[4], reverse=True)
            self.last_face_box = (candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3])
            return candidates

        # Do NOT generate fake face when no face exists in viewport
        self.last_face_box = None
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

        # If no face is detected in the frame, return cleanly with face_detected: False
        if not faces:
            return self._empty_features(now, "No astronaut face detected in camera viewport", baseline)

        # 3. Assess Face Quality
        face_quality = self.assess_face_quality(frame, gray, faces)

        # If quality is invalid due to extreme blur / lighting
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
        # Eye Region Analysis (Upper 55% of face)
        upper_face = face_roi[int(rh * 0.15):int(rh * 0.55), :]
        ear = baseline["resting_ear"]
        is_eye_closed = False

        if self.eye_cascade is not None and not self.eye_cascade.empty() and upper_face.size > 0:
            eyes = self.eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(rw * 0.10), int(rh * 0.08))
            )
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes_sorted[0], eyes_sorted[-1]
                ear_l = float(left_eye[3]) / max(1.0, float(left_eye[2]))
                ear_r = float(right_eye[3]) / max(1.0, float(right_eye[2]))
                ear = float(np.clip((ear_l + ear_r) / 2.0, 0.10, 0.45))
            elif len(eyes) == 1:
                ear = float(np.clip(eyes[0][3] / max(1.0, float(eyes[0][2])), 0.10, 0.45))
            else:
                # Eye cascade did not find open eyes — check if closed or low-light
                eye_roi_sample = face_roi[int(rh * 0.22):int(rh * 0.46), int(rw * 0.15):int(rw * 0.85)]
                if eye_roi_sample.size > 0:
                    eye_var = float(cv2.Sobel(eye_roi_sample, cv2.CV_64F, 0, 1, ksize=3).var())
                    # Very low vertical edge variance in eye strip indicates closed eyelids
                    if eye_var < 1500.0:
                        ear = 0.12
                        is_eye_closed = True
                    else:
                        ear = 0.26
        else:
            # Fallback estimation
            eye_roi_sample = face_roi[int(rh * 0.22):int(rh * 0.46), int(rw * 0.15):int(rw * 0.85)]
            if eye_roi_sample.size > 0:
                eye_edges = cv2.Canny(eye_roi_sample, 40, 120)
                density = float(np.sum(eye_edges > 0) / max(1.0, eye_edges.size))
                ear = float(np.clip(0.16 + (density * 0.8), 0.12, 0.42))

        # True eye closure requires EAR < 0.15
        if ear < 0.15:
            is_eye_closed = True

        self.ear_history.append((now, ear))
        # Blink detection
        if len(self.ear_history) >= 2 and ear < 0.16 and self.ear_history[-2][1] >= 0.16:
            self.blink_timestamps.append(now)

        # Mouth Region Analysis (Lower 45% of face)
        lower_face = face_roi[int(rh * 0.55):int(rh * 0.96), int(rw * 0.12):int(rw * 0.88)]
        mar = baseline["resting_mar"]
        smile_detected = False
        smile_intensity = 0.0
        frown_intensity = 0.0
        is_yawning = False

        if lower_face.size > 0:
            # 1. Haar Smile Detector
            if self.smile_cascade is not None and not self.smile_cascade.empty():
                smiles = self.smile_cascade.detectMultiScale(
                    lower_face,
                    scaleFactor=1.18,
                    minNeighbors=6,
                    minSize=(int(rw * 0.18), int(rh * 0.08))
                )
                if len(smiles) > 0:
                    smile_detected = True
                    smile_intensity = float(min(1.0, 0.50 + len(smiles) * 0.25))

            # 2. Geometric Mouth Analysis (Otsu segmentation)
            _, mouth_thresh = cv2.threshold(lower_face, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mouth_vert = np.sum(mouth_thresh, axis=1) / max(1.0, lower_face.shape[1])
            mouth_h = np.count_nonzero(mouth_vert > 20)
            mouth_horiz = np.sum(mouth_thresh, axis=0) / max(1.0, lower_face.shape[0])
            mouth_w = np.count_nonzero(mouth_horiz > 15)

            lh, lw = lower_face.shape[:2]
            mar = float(np.clip(mouth_h / max(1.0, float(mouth_w)), 0.10, 0.85))
            mw_ratio = float(mouth_w) / max(1.0, float(rw))

            # Smile expands mouth horizontally (>0.44 of face width) with moderate openness
            if mw_ratio > 0.44 and mar < 0.50:
                geom_smile = float(np.clip((mw_ratio - 0.42) * 4.0, 0.0, 1.0))
                smile_intensity = max(smile_intensity, geom_smile)
                if geom_smile > 0.35:
                    smile_detected = True

            # Frown (AU15 / Lip Corner Depressor): Mouth compressed downward with narrow width and corners drooping
            if not smile_detected and mw_ratio < 0.40 and mar < 0.22:
                # Check bottom row variance indicating downward turned lips
                lower_lip_strip = mouth_thresh[int(lh * 0.6):, :]
                if lower_lip_strip.size > 0:
                    frown_intensity = float(np.clip(0.35 + (0.22 - mar) * 2.5, 0.0, 0.90))

            # Yawn Detection: Sustained wide mouth opening (MAR > 0.68) without smiling
            if not smile_detected and mar > 0.68:
                is_yawning = True
                if not self.yawn_timestamps or (now - self.yawn_timestamps[-1] > 4.0):
                    self.yawn_timestamps.append(now)

        self.mar_history.append((now, mar))

        # Brow Furrow / Glabella Strain (AU04)
        glabella_roi = face_roi[int(rh * 0.18):int(rh * 0.38), int(rw * 0.35):int(rw * 0.65)]
        brow_furrow = baseline["resting_au04"]
        if glabella_roi.size > 0:
            sobelx = cv2.Sobel(glabella_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(glabella_roi, cv2.CV_64F, 0, 1, ksize=3)
            glabella_var = float(np.var(sobelx) + np.var(sobely))
            # Intense forehead creases elevate variance above baseline
            brow_furrow = float(np.clip((glabella_var - 6000.0) / 28000.0, 0.05, 0.95))

        # Purge rolling buffers older than 60s
        cutoff = now - 60.0
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff]
        self.yawn_timestamps = [t for t in self.yawn_timestamps if t >= cutoff]
        self.ear_history = [(t, v) for t, v in self.ear_history if t >= cutoff]
        self.mar_history = [(t, v) for t, v in self.mar_history if t >= cutoff]

        # PERCLOS: Only meaningful with at least 15 rolling frames to avoid false spikes
        if len(self.ear_history) >= 15:
            closed_count = sum(1 for _, e in self.ear_history if e < 0.15)
            perclos = float(closed_count / len(self.ear_history))
        else:
            perclos = 0.0

        blinks_per_min = float(len(self.blink_timestamps))
        yawns_per_min = len(self.yawn_timestamps)

        # Baseline comparisons
        base_ear = baseline["resting_ear"]
        base_mar = baseline["resting_mar"]
        base_blinks = baseline["blink_rate_bpm"]
        base_au04 = baseline["resting_au04"]

        delta_ear = ear - base_ear
        delta_mar = mar - base_mar
        delta_blinks = blinks_per_min - base_blinks
        delta_au04 = brow_furrow - base_au04

        # -------------------------------------------------------------
        # 5. FACS Logits for All 7 Discrete Emotion Classes
        # -------------------------------------------------------------
        logits: Dict[str, float] = {}

        # 1. Happy: Smile AU12, elevated lip corners, cheek raise AU06
        logits["happy"] = (smile_intensity * 4.5) + (1.2 if smile_detected else 0.0) - (brow_furrow * 2.2) - (perclos * 3.0)

        # 2. Stressed: Brow furrow AU04, eye narrowing/tension (0.16-0.23), lip stretch AU20
        au20_stretch = float(np.clip(mar * 0.8, 0.0, 1.0))
        logits["stressed"] = (brow_furrow * 3.6) + (au20_stretch * 1.8) + (0.8 if (0.16 <= ear <= 0.23 and mar < 0.25) else 0.0) - (smile_intensity * 3.5)

        # 3. Frustrated: Intense brow furrow AU04 + jaw clenching / compressed mouth
        logits["frustrated"] = (brow_furrow * 4.2) + (1.2 if mar < 0.16 and brow_furrow > 0.35 else 0.0) - (smile_intensity * 4.0)

        # 4. Sad: Frown AU15 (corners down), drooping affect, low energy
        logits["sad"] = (frown_intensity * 3.8) + (0.8 if mar < 0.18 and brow_furrow > 0.20 and smile_intensity < 0.1 else 0.0) - (smile_intensity * 4.0)

        # 5. Anxious: Wide open eyes (high EAR > 0.30), open mouth / gasp (MAR > 0.26), brow tension
        wide_eye = max(0.0, (ear - 0.28) * 14.0)
        open_mouth = max(0.0, (mar - 0.26) * 8.0)
        logits["anxious"] = wide_eye + open_mouth + (brow_furrow * 1.2) - (smile_intensity * 2.5)

        # 6. Fatigued: True eye closure (EAR < 0.15), valid PERCLOS > 0.25, or sustained yawn
        eye_closure_fatigue = 3.8 if is_eye_closed else 0.0
        perclos_fatigue = (perclos * 4.0) if (len(self.ear_history) >= 15 and perclos > 0.25) else 0.0
        yawn_fatigue = 3.2 if is_yawning else 0.0
        logits["fatigued"] = eye_closure_fatigue + perclos_fatigue + yawn_fatigue - (smile_intensity * 3.0)

        # 7. Neutral: Baseline when action units are relaxed and resting
        logits["neutral"] = 2.2 - (smile_intensity * 3.0) - (brow_furrow * 2.8) - (frown_intensity * 2.8) - (wide_eye * 2.5) - (eye_closure_fatigue * 2.0) - (perclos_fatigue * 2.0)

        # Softmax Normalization
        exp_logits = {k: np.exp(np.clip(v, -5.0, 5.0)) for k, v in logits.items()}
        total_exp = sum(exp_logits.values())
        norm_frame_probs = {k: float(v / total_exp) for k, v in exp_logits.items()}

        # -------------------------------------------------------------
        # 6. Temporal Smoothing & State Hysteresis
        # -------------------------------------------------------------
        # Fast responsive alpha = 0.45 (~0.5s transition time)
        alpha = 0.45
        for c in self.classes:
            self.ema_probabilities[c] = (alpha * norm_frame_probs[c]) + ((1.0 - alpha) * self.ema_probabilities[c])

        ema_tot = sum(self.ema_probabilities.values())
        smoothed_probs = {k: round(v / ema_tot, 4) for k, v in self.ema_probabilities.items()}

        # Leading candidate
        leading_candidate = max(smoothed_probs.items(), key=lambda item: item[1])[0]

        # State Hysteresis: Require consecutive candidate persistence
        if leading_candidate == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = leading_candidate
            self.candidate_count = 1

        if self.candidate_count >= self.required_persistence_frames and smoothed_probs[leading_candidate] > 0.32:
            self.stable_dominant_emotion = leading_candidate
            self.stable_facial_state = self.state_labels.get(leading_candidate, "relaxed")

        self.recent_state_history.append(self.stable_dominant_emotion)
        if len(self.recent_state_history) > 8:
            self.recent_state_history.pop(0)

        consistency = float(self.recent_state_history.count(self.stable_dominant_emotion)) / len(self.recent_state_history)
        final_confidence = float(np.clip(smoothed_probs[self.stable_dominant_emotion] * 0.65 + consistency * 0.25 + face_quality["quality_score"] * 0.10, 0.45, 0.98))

        # Valence & Arousal
        valence = (smoothed_probs.get("happy", 0.0) * 0.9) - (smoothed_probs.get("stressed", 0.0) * 0.6) - (smoothed_probs.get("fatigued", 0.0) * 0.4) - (smoothed_probs.get("frustrated", 0.0) * 0.7) - (smoothed_probs.get("sad", 0.0) * 0.8)
        arousal = (smoothed_probs.get("stressed", 0.0) * 0.8) + (smoothed_probs.get("anxious", 0.0) * 0.85) + (smoothed_probs.get("happy", 0.0) * 0.5) + (smoothed_probs.get("frustrated", 0.0) * 0.75) - (smoothed_probs.get("fatigued", 0.0) * 0.6) - (smoothed_probs.get("sad", 0.0) * 0.4)

        stress_indicator = float(np.clip((brow_furrow * 0.6) + (smoothed_probs.get("stressed", 0.0) * 0.4), 0.0, 1.0))
        fatigue_indicator = float(np.clip((perclos * 2.0) + (1.0 if is_eye_closed else 0.0) * 0.5 + (0.5 if is_yawning else 0.0), 0.0, 1.0))

        action_units = {
            "AU04_brow_furrow": round(brow_furrow, 3),
            "AU06_cheek_raiser": round(smile_intensity * 0.75, 3),
            "AU12_lip_corner_puller": round(smile_intensity, 3),
            "AU15_lip_corner_depressor": round(frown_intensity, 3),
            "AU20_lip_stretcher": round(float(np.clip(mar * 0.8, 0.0, 1.0)), 3),
            "AU25_lips_part": round(float(1.0 if mar > 0.35 else mar / 0.35), 3),
            "AU43_eye_closure": round(float(1.0 if is_eye_closed else 0.0), 3)
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
                "frown_intensity_au15": round(frown_intensity, 3),
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
        """Gracefully hold standby state when optical quality is degraded or no face is found."""
        is_no_face = face_quality.get("status") == "NO_FACE"
        return {
            "astronaut_id": self.active_astronaut_id,
            "facial_state": "standby" if is_no_face else self.stable_facial_state,
            "dominant_emotion": "standby" if is_no_face else self.stable_dominant_emotion,
            "stress_indicator": 0.0,
            "fatigue_indicator": 0.0,
            "facial_indicators": {
                "eye_aspect_ratio": baseline["resting_ear"],
                "mouth_aspect_ratio": baseline["resting_mar"],
                "brow_tension_au04": baseline["resting_au04"],
                "smile_intensity_au12": 0.0,
                "frown_intensity_au15": 0.0,
                "blink_rate_bpm": baseline["blink_rate_bpm"],
                "perclos": 0.0,
                "yawns_per_min": 0
            },
            "confidence": 0.0 if is_no_face else 0.30,
            "face_quality": face_quality,
            "baseline_comparison": {
                "baseline_ear": baseline["resting_ear"],
                "ear_deviation": 0.0,
                "baseline_mar": baseline["resting_mar"],
                "mar_deviation": 0.0,
                "baseline_blinks": baseline["blink_rate_bpm"],
                "blink_deviation": 0.0
            },
            "face_detected": not is_no_face,
            "face_count": 0 if is_no_face else 1,
            "multiple_faces": False,
            "face_bounding_box": None if is_no_face else self.last_face_box,
            "lighting": face_quality,
            "eye_aspect_ratio": baseline["resting_ear"],
            "mouth_aspect_ratio": baseline["resting_mar"],
            "blinks_per_min": baseline["blink_rate_bpm"],
            "yawns_per_min": 0,
            "perclos": 0.0,
            "action_units": {au: 0.0 for au in ["AU04_brow_furrow", "AU06_cheek_raiser", "AU12_lip_corner_puller", "AU15_lip_corner_depressor", "AU20_lip_stretcher", "AU25_lips_part", "AU43_eye_closure"]},
            "probabilities": {c: (1.0 if c == 'neutral' else 0.0) for c in self.classes},
            "valence": 0.0,
            "arousal": 0.0,
            "modality_active": not is_no_face,
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

