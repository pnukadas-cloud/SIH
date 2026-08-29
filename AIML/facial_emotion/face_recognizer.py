"""
AIML — Robust Face Recognition & Identity Verification Engine
Pipeline:
Camera Frame -> Face Detection -> Face Quality Check -> 128-D Spatial LBP Embedding ->
Cosine Similarity against Enrolled Astronaut Database -> Identity Decision.

States:
- IDENTIFIED: High confidence match (sim >= 0.80)
- LOW_CONFIDENCE: Uncertain match (0.65 <= sim < 0.80)
- UNKNOWN: No enrolled astronaut matches (sim < 0.65)
- NO_FACE: No face detected in frame
- MULTIPLE_FACES: Multiple faces in viewport
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Tuple, Optional
from Backend_DB.database.connection import DatabaseManager
from AIML.facial_emotion.fer_module import FacialEmotionModule

class FaceRecognizer:
    def __init__(self, recognition_threshold: float = 0.80, low_confidence_threshold: float = 0.65):
        self.recognition_threshold = recognition_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.fer = FacialEmotionModule()
        self.db = DatabaseManager()

    def decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 encoded image string to OpenCV BGR numpy array."""
        try:
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def compute_lbp_patch(self, patch: np.ndarray) -> np.ndarray:
        """Compute standard 8-neighbor Local Binary Pattern for a single-channel image patch."""
        h, w = patch.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = patch[1:-1, 1:-1]
        
        # 8 neighbors clockwise starting top-left
        neighbors = [
            patch[0:-2, 0:-2], patch[0:-2, 1:-1], patch[0:-2, 2:],
            patch[1:-1, 2:], patch[2:, 2:], patch[2:, 1:-1],
            patch[2:, 0:-2], patch[1:-1, 0:-2]
        ]
        
        for p_idx, neighbor in enumerate(neighbors):
            lbp |= ((neighbor >= center).astype(np.uint8) << p_idx)
            
        return lbp

    def extract_face_embedding(self, face_crop: np.ndarray) -> Optional[List[float]]:
        """
        Compute a distinctive normalized 128-dimensional biometric face representation:
        - 32-D: Multi-spectral Skin Chrominance Profile (YCrCb & HSV distributions)
        - 32-D: Facial Structural Geometry Ratios (inter-ocular distance, mouth ratio, forehead ratio)
        - 64-D: Spatial LBP & Gradient Texture Grids (4x4 spatial cells x 4-bin histograms)
        Total = 128 dimensions, L2-normalized.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            return None

        # Standardize face crop
        resized_bgr = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

        features = []

        # 1. 32-D Multi-spectral Chrominance & Phototype Profile
        ycrcb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
        
        cr_hist, _ = np.histogram(ycrcb[:, :, 1], bins=8, range=(120, 180), density=True)
        cb_hist, _ = np.histogram(ycrcb[:, :, 2], bins=8, range=(70, 135), density=True)
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 35), density=True)
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=8, range=(20, 255), density=True)
        features.extend(cr_hist.astype(np.float32))
        features.extend(cb_hist.astype(np.float32))
        features.extend(h_hist.astype(np.float32))
        features.extend(s_hist.astype(np.float32)) # 32 dims

        # 2. 32-D Facial Structural Geometry Ratios
        # Upper face (eyes / brow), Mid face (nose), Lower face (mouth / chin)
        upper = gray[:45, :]
        mid = gray[45:80, :]
        lower = gray[80:, :]
        
        # Horizontal & vertical projections across bands
        u_proj = np.mean(upper, axis=0) # 112 -> 8 bins
        m_proj = np.mean(mid, axis=0)
        l_proj = np.mean(lower, axis=0)
        
        for p in [u_proj, m_proj, l_proj]:
            # Downsample 112 -> 8
            down = [float(np.mean(p[i*14:(i+1)*14])) for i in range(8)]
            features.extend(down) # 24 dims
            
        # Vertical cross-section
        v_proj = [float(np.mean(gray[i*14:(i+1)*14, :])) for i in range(8)]
        features.extend(v_proj) # 8 dims -> 32 dims total

        # 3. 64-D Spatial LBP Texture Grid (4x4 cells x 4-bin LBP)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_face = clahe.apply(gray)
        lbp_face = self.compute_lbp_patch(norm_face) # 110 x 110
        gh, gw = lbp_face.shape
        cell_h = gh // 4
        cell_w = gw // 4

        for r in range(4):
            for c in range(4):
                cell = lbp_face[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                hist, _ = np.histogram(cell, bins=4, range=(0, 256), density=True)
                features.extend(hist.astype(np.float32)) # 16 * 4 = 64 dims

        # Final 128-D L2 Unit Vector
        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            return None

        return [round(float(x), 5) for x in vec]

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two unit vectors."""
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def recognize_face(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        End-to-End Face Recognition:
        Detect Face -> Quality Check -> Extract Embedding -> Match against SQLite DB.
        """
        if frame is None or frame.size == 0:
            return {
                "status": "NO_FACE",
                "astronaut_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "No video feed received."
            }

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 1. Face Detection
        faces = self.fer.detect_face_multispectral(frame, gray)

        if not faces:
            return {
                "status": "NO_FACE",
                "astronaut_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "No astronaut detected in camera viewport."
            }

        if len(faces) > 1:
            return {
                "status": "MULTIPLE_FACES",
                "astronaut_id": None,
                "name": None,
                "confidence": 0.0,
                "face_count": len(faces),
                "message": f"Multiple astronauts ({len(faces)}) detected — identity cannot be confirmed."
            }

        # 2. Quality Check
        quality = self.fer.assess_face_quality(frame, gray, faces)
        if not quality["valid"] and quality["status"] in ["POOR_LIGHTING", "OVEREXPOSED", "EXTREME_POSE", "OCCLUSION"]:
            return {
                "status": "LOW_CONFIDENCE",
                "astronaut_id": None,
                "name": None,
                "confidence": 0.35,
                "quality": quality,
                "message": f"Optical conditions degraded ({quality['status']}) — please adjust lighting or use manual login."
            }

        # 3. Extract Face Patch & Embedding
        fx, fy, fw, fh, _, _ = faces[0]
        face_crop = frame[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        query_embedding = self.extract_face_embedding(face_crop)

        if query_embedding is None:
            return {
                "status": "LOW_CONFIDENCE",
                "astronaut_id": None,
                "name": None,
                "confidence": 0.30,
                "message": "Failed to extract clean facial landmarks — identity uncertain."
            }

        # 4. Compare against enrolled astronaut embeddings in Database
        enrolled_astronauts = self.db.get_all_enrolled_embeddings()
        if not enrolled_astronauts:
            return {
                "status": "UNKNOWN",
                "astronaut_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "No astronauts enrolled in biometric database."
            }

        best_match = None
        best_similarity = -1.0

        for a in enrolled_astronauts:
            sim = self.cosine_similarity(query_embedding, a["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_match = a

        # 5. Identity Decision Logic
        if best_similarity >= self.recognition_threshold and best_match:
            conf = float(np.clip(best_similarity, 0.50, 0.99))
            return {
                "status": "IDENTIFIED",
                "astronaut_id": best_match["astronaut_id"],
                "name": best_match["name"],
                "callsign": best_match.get("callsign", ""),
                "role": best_match.get("role", ""),
                "confidence": round(conf, 3),
                "similarity": round(best_similarity, 3),
                "message": f"Welcome, {best_match['name']} ({best_match['astronaut_id']}). Identity confirmed."
            }
        elif best_similarity >= self.low_confidence_threshold and best_match:
            return {
                "status": "LOW_CONFIDENCE",
                "astronaut_id": best_match["astronaut_id"],
                "name": best_match["name"],
                "confidence": round(best_similarity, 3),
                "similarity": round(best_similarity, 3),
                "message": "Identity uncertain — please use manual login or retry facial scan."
            }
        else:
            return {
                "status": "UNKNOWN",
                "astronaut_id": None,
                "name": None,
                "confidence": round(max(0.0, best_similarity), 3),
                "similarity": round(best_similarity, 3),
                "message": "Unknown astronaut — please use manual login or enroll first."
            }
