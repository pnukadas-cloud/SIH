"""
AIML — Facial Emotion Recognition (FER) Module
Extracts authentic geometric facial biomarkers, checks environmental conditions
(lighting, face count), and calculates FACS-based emotion probabilities.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, List, Tuple

class FacialEmotionModule:
    def __init__(self):
        self.classes = ['neutral', 'happy', 'stressed', 'fatigued', 'anxious', 'sad', 'frustrated']
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        
        # Load OpenCV Haar cascades if available
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            except Exception:
                pass

        # Rolling temporal buffers
        self.blink_timestamps: List[float] = []
        self.yawn_timestamps: List[float] = []
        self.ear_history: List[Tuple[float, float]] = []
        self.mar_history: List[Tuple[float, float]] = []

    def check_lighting(self, gray: np.ndarray) -> Dict[str, Any]:
        """Evaluate ambient cabin lighting and image sharpness."""
        mean_brightness = float(np.mean(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 0 else 0.0
        
        if mean_brightness < 35:
            lighting_status = "POOR_LIGHTING"
            lighting_msg = "Low cabin illumination detected; optical features degraded."
        elif mean_brightness > 230:
            lighting_status = "OVEREXPOSED"
            lighting_msg = "Severe optical glare or overexposure detected."
        else:
            lighting_status = "OPTIMAL"
            lighting_msg = "Cabin lighting nominal for facial landmark extraction."
            
        is_blurred = laplacian_var < 20.0
        return {
            "brightness": round(mean_brightness, 1),
            "sharpness": round(laplacian_var, 1),
            "status": lighting_status,
            "message": lighting_msg,
            "is_blurred": is_blurred
        }

    def detect_faces(self, gray: np.ndarray, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect all face bounding boxes in frame."""
        h, w = gray.shape[:2]
        
        # Primary: Haar Cascade
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(w * 0.12), int(h * 0.12))
            )
            if len(faces) > 0:
                return [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces]
                
        # Secondary: HSV Skin Chrominance Segmentation Fallback
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 20, 70], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected = []
            for c in contours:
                if cv2.contourArea(c) > (w * h * 0.05):
                    fx, fy, fw, fh = cv2.boundingRect(c)
                    if 0.8 <= (fh / max(1.0, float(fw))) <= 1.8:
                        detected.append((int(fx), int(fy), int(fw), int(fh)))
            if detected:
                return detected
                
        return []

    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract authentic geometric Action Units and ocular biometrics."""
        now = time.time()
        if frame is None or frame.size == 0:
            return self._empty_features(now, "No video stream received")

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lighting = self.check_lighting(gray)
        faces = self.detect_faces(gray, frame)
        
        num_faces = len(faces)
        if num_faces == 0:
            return {
                **self._empty_features(now, "No face detected in optical field of view"),
                "lighting": lighting,
                "face_count": 0
            }

        # Select primary (largest) face
        primary_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = primary_face
        face_roi = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        
        # 1. Eye Aspect Ratio (EAR) & Blink Tracking
        upper_face = face_roi[int(fh * 0.15):int(fh * 0.55), :]
        ear = 0.28
        if self.eye_cascade is not None and not self.eye_cascade.empty() and upper_face.size > 0:
            eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=4, minSize=(int(fw * 0.10), int(fh * 0.07)))
            if len(eyes) >= 2:
                sorted_eyes = sorted(eyes, key=lambda e: e[0])
                ear1 = float(sorted_eyes[0][3]) / max(1.0, float(sorted_eyes[0][2]))
                ear2 = float(sorted_eyes[-1][3]) / max(1.0, float(sorted_eyes[-1][2]))
                ear = float(np.clip((ear1 + ear2) / 2.0, 0.12, 0.45))
            elif len(eyes) == 1:
                ear = float(np.clip(eyes[0][3] / max(1.0, float(eyes[0][2])), 0.12, 0.45))
        else:
            # Gradient intensity fallback
            if upper_face.size > 0:
                edges = cv2.Canny(upper_face, 50, 150)
                density = float(np.sum(edges > 0) / max(1.0, edges.size))
                ear = float(np.clip(0.18 + (density * 1.1), 0.14, 0.38))

        self.ear_history.append((now, ear))
        if len(self.ear_history) >= 2 and ear < 0.20 and self.ear_history[-2][1] >= 0.20:
            self.blink_timestamps.append(now)

        # 2. Mouth Aspect Ratio (MAR) & Yawn Tracking
        lower_face = face_roi[int(fh * 0.55):int(fh * 0.95), int(fw * 0.15):int(fw * 0.85)]
        mar = 0.22
        if lower_face.size > 0:
            _, mthresh = cv2.threshold(lower_face, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(mthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_m = max(contours, key=cv2.contourArea)
                _, _, mw, mh = cv2.boundingRect(largest_m)
                mar = float(np.clip(mh / max(1.0, float(mw)), 0.10, 0.85))

        self.mar_history.append((now, mar))
        if mar > 0.52 and (not self.yawn_timestamps or (now - self.yawn_timestamps[-1] > 3.0)):
            self.yawn_timestamps.append(now)

        # 3. Smile Intensity / AU12
        smile_intensity = 0.0
        if self.smile_cascade is not None and not self.smile_cascade.empty() and lower_face.size > 0:
            smiles = self.smile_cascade.detectMultiScale(lower_face, scaleFactor=1.2, minNeighbors=8)
            smile_intensity = float(min(1.0, len(smiles) * 0.5))
        else:
            smile_intensity = float(np.clip((mar - 0.25) * 2.0, 0.0, 1.0)) if mar > 0.25 else 0.1

        # Purge rolling buffers older than 60 seconds
        cutoff = now - 60.0
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff]
        self.yawn_timestamps = [t for t in self.yawn_timestamps if t >= cutoff]
        self.ear_history = [(t, v) for t, v in self.ear_history if t >= cutoff]
        self.mar_history = [(t, v) for t, v in self.mar_history if t >= cutoff]

        perclos = float(sum(1 for _, e in self.ear_history if e < 0.21) / max(1, len(self.ear_history)))

        # Facial Action Units (FACS)
        action_units = {
            "AU04_brow_furrow": float(np.clip(1.0 - (ear * 2.2), 0.0, 1.0)),
            "AU06_cheek_raiser": float(smile_intensity * 0.7),
            "AU12_lip_corner_puller": smile_intensity,
            "AU20_lip_stretcher": float(np.clip(mar * 0.9, 0.0, 1.0)),
            "AU25_lips_part": float(1.0 if mar > 0.35 else mar / 0.35),
            "AU43_eye_closure": float(1.0 if ear < 0.20 else 0.0)
        }

        # Calculate FACS logits
        logits = {
            "happy": (action_units["AU12_lip_corner_puller"] * 4.0) + (1.2 if smile_intensity > 0.3 else 0.0) - (action_units["AU04_brow_furrow"] * 2.0),
            "fatigued": (perclos * 4.5) + (len(self.yawn_timestamps) * 1.5) + (action_units["AU43_eye_closure"] * 2.0),
            "stressed": (action_units["AU04_brow_furrow"] * 3.5) + (action_units["AU20_lip_stretcher"] * 1.8) - (action_units["AU12_lip_corner_puller"] * 1.5),
            "anxious": (action_units["AU04_brow_furrow"] * 2.5) + (action_units["AU20_lip_stretcher"] * 2.5) + (1.5 if ear > 0.35 else 0.0),
            "sad": (action_units["AU04_brow_furrow"] * 2.0) + (1.2 if mar < 0.18 else 0.0) + (perclos * 1.2) - (action_units["AU12_lip_corner_puller"] * 3.0),
            "frustrated": (action_units["AU04_brow_furrow"] * 3.8) + (action_units["AU20_lip_stretcher"] * 2.2) - (action_units["AU12_lip_corner_puller"] * 2.5),
            "neutral": 1.2 - (action_units["AU04_brow_furrow"] * 1.5) - (action_units["AU12_lip_corner_puller"] * 1.5)
        }

        # Softmax normalization
        exp_logits = {k: np.exp(np.clip(v, -5.0, 5.0)) for k, v in logits.items()}
        total_exp = sum(exp_logits.values())
        probabilities = {k: round(float(v / total_exp), 4) for k, v in exp_logits.items()}
        dominant = max(probabilities.items(), key=lambda x: x[1])

        # Authentic Valence (-1.0 to +1.0) and Arousal (0.0 to 1.0)
        valence = (probabilities["happy"] * 1.0) - (probabilities["sad"] * 0.8) - (probabilities["frustrated"] * 0.7) - (probabilities["stressed"] * 0.6) - (probabilities["anxious"] * 0.6) - (probabilities["fatigued"] * 0.4)
        arousal = (probabilities["frustrated"] * 0.9) + (probabilities["anxious"] * 0.85) + (probabilities["stressed"] * 0.75) + (probabilities["happy"] * 0.6) + (probabilities["neutral"] * 0.2) + (probabilities["sad"] * 0.3) + (probabilities["fatigued"] * 0.1)

        # Confidence based on lighting, face size, and sharpness
        quality_factor = 1.0 if lighting["status"] == "OPTIMAL" else 0.7
        size_factor = min(1.0, (fw * fh) / (w * h * 0.15))
        final_confidence = round(float(dominant[1] * quality_factor * size_factor), 3)

        return {
            "face_detected": True,
            "face_count": num_faces,
            "multiple_faces": num_faces > 1,
            "face_bounding_box": {"x": fx, "y": fy, "w": fw, "h": fh},
            "lighting": lighting,
            "eye_aspect_ratio": round(ear, 3),
            "mouth_aspect_ratio": round(mar, 3),
            "blinks_per_min": float(len(self.blink_timestamps)),
            "yawns_per_min": len(self.yawn_timestamps),
            "perclos": round(perclos, 3),
            "action_units": {k: round(v, 3) for k, v in action_units.items()},
            "probabilities": probabilities,
            "dominant_emotion": dominant[0],
            "confidence": final_confidence,
            "valence": round(float(np.clip(valence, -1.0, 1.0)), 3),
            "arousal": round(float(np.clip(arousal, 0.0, 1.0)), 3),
            "modality_active": True,
            "timestamp": now
        }

    def _empty_features(self, timestamp: float, reason: str) -> Dict[str, Any]:
        uniform_probs = {c: round(1.0 / len(self.classes), 4) for c in self.classes}
        uniform_probs["neutral"] = 0.40
        tot = sum(uniform_probs.values())
        probs = {k: round(v / tot, 4) for k, v in uniform_probs.items()}
        return {
            "face_detected": False,
            "face_count": 0,
            "multiple_faces": False,
            "face_bounding_box": None,
            "lighting": {"status": "UNKNOWN", "message": reason, "brightness": 0, "sharpness": 0},
            "eye_aspect_ratio": 0.28,
            "mouth_aspect_ratio": 0.20,
            "blinks_per_min": 0.0,
            "yawns_per_min": 0,
            "perclos": 0.0,
            "action_units": {au: 0.0 for au in ["AU04_brow_furrow", "AU06_cheek_raiser", "AU12_lip_corner_puller", "AU20_lip_stretcher", "AU25_lips_part", "AU43_eye_closure"]},
            "probabilities": probs,
            "dominant_emotion": "neutral",
            "confidence": 0.25,
            "valence": 0.0,
            "arousal": 0.15,
            "modality_active": False,
            "reason": reason,
            "timestamp": timestamp
        }
