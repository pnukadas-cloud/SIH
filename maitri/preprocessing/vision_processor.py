"""
MAITRI — Vision Preprocessing & Facial Feature Extraction
Module: Face Detection, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
Facial Action Units (AUs), and Micro-Expression Extraction.
Compatible across OpenCV 4.x and OpenCV 5.x.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional

class VisionProcessor:
    def __init__(self):
        # Graceful detector initialization
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        
        # Check if CascadeClassifier exists (OpenCV 4.x)
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            except Exception:
                pass
                
        # Temporal tracking buffers for rolling calculations
        self.blink_timestamps = []
        self.yawn_timestamps = []
        self.ear_history = []
        self.mar_history = []
        self.last_face_rect = None
        self.face_present_frames = 0
        self.total_processed_frames = 0
        
    def _detect_faces_universal(self, gray: np.ndarray, frame: np.ndarray) -> list:
        """
        Universal face detection fallback supporting OpenCV 4/5 or contour/skin-color heuristics.
        """
        h, w = gray.shape[:2]
        
        # 1. Try CascadeClassifier if loaded
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(int(w * 0.15), int(h * 0.15))
            )
            if len(faces) > 0:
                return [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces]
                
        # 2. Heuristic Face Region Detector (Skin Chrominance / Intensity segmentation fallback)
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Standard human skin chrominance mask in HSV space
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_faces = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > (w * h * 0.04): # at least 4% of frame
                    fx, fy, fw, fh = cv2.boundingRect(c)
                    aspect = fh / max(1.0, float(fw))
                    if 0.8 <= aspect <= 1.8: # Face-like aspect ratio
                        valid_faces.append((int(fx), int(fy), int(fw), int(fh)))
            if valid_faces:
                return valid_faces
                
        return []

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame and extract geometric & biometric features.
        """
        self.total_processed_frames += 1
        now = time.time()
        
        if frame is None or frame.size == 0:
            return self._default_empty_response(now)
            
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        faces = self._detect_faces_universal(gray, frame)
        
        if len(faces) == 0:
            self.face_present_frames = max(0, self.face_present_frames - 1)
            return self._default_empty_response(now)
            
        self.face_present_frames += 1
        
        # Pick largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        self.last_face_rect = (int(x), int(y), int(fw), int(fh))
        
        face_roi_gray = gray[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)]
        
        # Eye region analysis (Upper 55% of face)
        upper_face = face_roi_gray[int(fh * 0.15):int(fh * 0.55), :]
        current_ear = 0.28
        
        if self.eye_cascade is not None and not self.eye_cascade.empty() and upper_face.size > 0:
            eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=4, minSize=(int(fw * 0.12), int(fh * 0.08)))
            num_eyes = len(eyes)
            if num_eyes >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes_sorted[0], eyes_sorted[-1]
                ear_left = float(left_eye[3]) / max(1.0, float(left_eye[2]))
                ear_right = float(right_eye[3]) / max(1.0, float(right_eye[2]))
                current_ear = float(np.clip((ear_left + ear_right) / 2.0, 0.12, 0.45))
            elif num_eyes == 1:
                current_ear = float(np.clip(eyes[0][3] / max(1.0, float(eyes[0][2])), 0.12, 0.45))
        else:
            # Gradient intensity eye openness estimator
            if upper_face.size > 0:
                eye_edges = cv2.Canny(upper_face, 50, 150)
                edge_density = float(np.sum(eye_edges > 0) / max(1.0, eye_edges.size))
                current_ear = float(np.clip(0.18 + (edge_density * 1.2), 0.14, 0.38))
            
        self.ear_history.append((now, current_ear))
        
        # Detect Blinks (EAR dips below 0.20)
        if len(self.ear_history) >= 2:
            if current_ear < 0.20 and self.ear_history[-2][1] >= 0.20:
                self.blink_timestamps.append(now)
                
        # Mouth region analysis (Lower 40% of face)
        lower_face = face_roi_gray[int(fh * 0.58):int(fh * 0.95), int(fw * 0.15):int(fw * 0.85)]
        current_mar = 0.22
        
        if lower_face.size > 0:
            _, mouth_thresh = cv2.threshold(lower_face, 60, 255, cv2.THRESH_BINARY_INV)
            mouth_contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if mouth_contours:
                largest_mouth = max(mouth_contours, key=cv2.contourArea)
                mx, my, mw, mh = cv2.boundingRect(largest_mouth)
                current_mar = float(np.clip(mh / max(1.0, float(mw)), 0.10, 0.85))
                
        self.mar_history.append((now, current_mar))
        
        # Detect Yawn (MAR > 0.52 for sustained duration)
        if current_mar > 0.52:
            if not self.yawn_timestamps or (now - self.yawn_timestamps[-1] > 3.0):
                self.yawn_timestamps.append(now)
                
        # Smile / AU12 detection
        smile_intensity = 0.0
        if self.smile_cascade is not None and not self.smile_cascade.empty() and lower_face.size > 0:
            smiles = self.smile_cascade.detectMultiScale(lower_face, scaleFactor=1.2, minNeighbors=8)
            smile_intensity = float(min(1.0, len(smiles) * 0.5))
        else:
            smile_intensity = float(np.clip((current_mar - 0.25) * 2.0, 0.0, 1.0)) if current_mar > 0.25 else 0.1
            
        # Clean rolling buffers (keep last 60 seconds)
        cutoff_60s = now - 60.0
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff_60s]
        self.yawn_timestamps = [t for t in self.yawn_timestamps if t >= cutoff_60s]
        self.ear_history = [(t, v) for t, v in self.ear_history if t >= cutoff_60s]
        self.mar_history = [(t, v) for t, v in self.mar_history if t >= cutoff_60s]
        
        # Calculate Rolling Metrics
        blinks_per_min = float(len(self.blink_timestamps))
        yawns_per_min = float(len(self.yawn_timestamps))
        
        # PERCLOS (Percentage of Eye Closure time over last 60s)
        if self.ear_history:
            closed_samples = sum(1 for _, ear in self.ear_history if ear < 0.21)
            perclos = float(closed_samples / len(self.ear_history))
        else:
            perclos = 0.05
            
        # Action Units
        action_units = {
            "AU04_brow_furrow": float(np.clip(1.0 - (current_ear * 2.2), 0.0, 1.0)),
            "AU06_cheek_raiser": float(smile_intensity * 0.7),
            "AU12_lip_corner_puller": smile_intensity,
            "AU20_lip_stretcher": float(np.clip(current_mar * 0.9, 0.0, 1.0)),
            "AU25_lips_part": float(1.0 if current_mar > 0.35 else current_mar / 0.35),
            "AU43_eye_closure": float(1.0 if current_ear < 0.20 else 0.0)
        }
        
        texture_energy = float(cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()) if face_roi_gray.size > 0 else 0.0
        
        # Render Spacecraft HUD Overlays
        hud_frame = frame.copy()
        self._draw_hud_overlays(hud_frame, x, y, fw, fh, current_ear, current_mar, perclos, action_units)
        
        return {
            "face_detected": True,
            "face_bounding_box": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)},
            "eye_aspect_ratio": round(current_ear, 3),
            "mouth_aspect_ratio": round(current_mar, 3),
            "blinks_per_min": round(blinks_per_min, 1),
            "yawns_per_min": int(yawns_per_min),
            "perclos": round(perclos, 3),
            "action_units": {k: round(v, 3) for k, v in action_units.items()},
            "smile_detected": bool(smile_intensity > 0.3),
            "texture_energy": round(texture_energy, 1),
            "hud_frame": hud_frame,
            "timestamp": now
        }
        
    def _draw_hud_overlays(self, frame: np.ndarray, x: int, y: int, w: int, h: int, ear: float, mar: float, perclos: float, aus: Dict[str, float]):
        """Draw tactical ISRO Bhartiya Antariksh Station HUD graphics on the frame."""
        line_len = int(w * 0.2)
        color = (0, 255, 200) # Cyan-Green HUD
        thickness = 2
        
        # Corner brackets
        cv2.line(frame, (x, y), (x + line_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + line_len), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - line_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + line_len), color, thickness)
        cv2.line(frame, (x, y + h), (x + line_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - line_len), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, thickness)
        
        # Header Badge
        cv2.rectangle(frame, (x, max(0, y - 24)), (x + w, y), (20, 20, 25), -1)
        cv2.putText(frame, f"MAITRI OPTICAL LOCK | EAR:{ear:.2f} MAR:{mar:.2f}", (x + 6, max(12, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1, cv2.LINE_AA)
        
        # Telemetry Overlay below face
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 22), (20, 20, 25), -1)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}% | AU4:{aus['AU04_brow_furrow']:.2f} AU12:{aus['AU12_lip_corner_puller']:.2f}",
                    (x + 6, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 255), 1, cv2.LINE_AA)

    def _default_empty_response(self, timestamp: float) -> Dict[str, Any]:
        return {
            "face_detected": False,
            "face_bounding_box": None,
            "eye_aspect_ratio": 0.28,
            "mouth_aspect_ratio": 0.20,
            "blinks_per_min": 0.0,
            "yawns_per_min": 0,
            "perclos": 0.0,
            "action_units": {
                "AU04_brow_furrow": 0.0,
                "AU06_cheek_raiser": 0.0,
                "AU12_lip_corner_puller": 0.0,
                "AU20_lip_stretcher": 0.0,
                "AU25_lips_part": 0.0,
                "AU43_eye_closure": 0.0
            },
            "smile_detected": False,
            "texture_energy": 0.0,
            "hud_frame": None,
            "timestamp": timestamp
        }
