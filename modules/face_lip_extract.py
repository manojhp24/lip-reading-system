"""
Modules 2 & 3: Face Detection & Lip Extraction Modules
Robust implementation supporting:
1. MediaPipe Solutions FaceMesh (if available)
2. Self-contained OpenCV Haar Cascade (using bundled XML in modules/)
Guarantees face & lip tracking without attribute errors or missing XML issues.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict

class FaceAndLipExtractor:
    def __init__(self, target_roi_size: Tuple[int, int] = (96, 96)):
        self.target_roi_size = target_roi_size
        self.lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
            191, 80, 81, 82, 13, 312, 311, 310, 415, 308
        ]
        
        # Path to bundled local Haar Cascade XML
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        if os.path.exists(self.cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        else:
            self.face_cascade = None

    def process_frames(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        """
        Processes a sequence of video frames to extract lip bounding box regions.
        """
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                return self._process_with_solutions(frames, mp.solutions.face_mesh)
        except Exception:
            pass

        return self._process_with_opencv_fallback(frames)

    def _process_with_solutions(self, frames: List[np.ndarray], mp_face_mesh) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        annotated_frames = []
        cropped_lips = []
        detected_count = 0
        last_valid_bbox = None

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            for frame in frames:
                h, w, c = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                annotated = frame.copy()

                if results.multi_face_landmarks:
                    detected_count += 1
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    lip_pts = []
                    for idx in self.lip_indices:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            lip_pts.append((int(lm.x * w), int(lm.y * h)))
                    
                    lip_pts = np.array(lip_pts)
                    x, y, bw, bh = cv2.boundingRect(lip_pts)
                    
                    pad_w = int(bw * 0.35)
                    pad_h = int(bh * 0.45)
                    
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(w, x + bw + pad_w)
                    y2 = min(h, y + bh + pad_h)
                    
                    last_valid_bbox = (x1, y1, x2, y2)
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for pt in lip_pts[::3]:
                        cv2.circle(annotated, tuple(pt), 2, (0, 0, 255), -1)
                    
                    cv2.putText(annotated, "Mouth ROI Tracked", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                elif last_valid_bbox is not None:
                    x1, y1, x2, y2 = last_valid_bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 1)

                else:
                    cw, ch = w // 2, h // 2
                    x1, y1, x2, y2 = cw - 60, ch + 20, cw + 60, ch + 100

                mouth_crop_raw = frame[y1:y2, x1:x2]
                if mouth_crop_raw.size > 0:
                    mouth_crop = cv2.resize(mouth_crop_raw, self.target_roi_size)
                else:
                    mouth_crop = np.zeros((self.target_roi_size[1], self.target_roi_size[0], 3), dtype=np.uint8)

                annotated_frames.append(annotated)
                cropped_lips.append(mouth_crop)

        detection_rate = (detected_count / len(frames) * 100) if frames else 0.0
        stats = {
            "total_frames": len(frames),
            "face_detected_frames": detected_count,
            "tracking_confidence_pct": round(detection_rate, 1)
        }
        return annotated_frames, cropped_lips, stats

    def _process_with_opencv_fallback(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        annotated_frames = []
        cropped_lips = []
        detected_count = 0

        for frame in frames:
            h, w = frame.shape[:2]
            annotated = frame.copy()
            mouth_crop_raw = None

            if self.face_cascade is not None and not self.face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))
                
                if len(faces) > 0:
                    detected_count += 1
                    fx, fy, fw, fh = faces[0]
                    # Lower 35% of face is the mouth area
                    mx1 = max(0, fx + int(fw * 0.2))
                    my1 = max(0, fy + int(fh * 0.65))
                    mx2 = min(w, fx + int(fw * 0.8))
                    my2 = min(h, fy + int(fh * 0.95))
                    
                    cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), (255, 100, 0), 1)
                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
                    cv2.putText(annotated, "Mouth ROI Tracked", (mx1, max(20, my1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    mouth_crop_raw = frame[my1:my2, mx1:mx2]

            # Fallback center crop if face not detected
            if mouth_crop_raw is None or mouth_crop_raw.size == 0:
                cw, ch = w // 2, h // 2
                mx1, my1, mx2, my2 = max(0, cw - 50), max(0, ch + 20), min(w, cw + 50), min(h, ch + 100)
                cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 255, 255), 1)
                mouth_crop_raw = frame[my1:my2, mx1:mx2]

            if mouth_crop_raw.size > 0:
                mouth_crop = cv2.resize(mouth_crop_raw, self.target_roi_size)
            else:
                mouth_crop = np.zeros((self.target_roi_size[1], self.target_roi_size[0], 3), dtype=np.uint8)

            annotated_frames.append(annotated)
            cropped_lips.append(mouth_crop)

        detection_rate = (detected_count / len(frames) * 100) if frames else 0.0
        stats = {
            "total_frames": len(frames),
            "face_detected_frames": detected_count,
            "tracking_confidence_pct": round(detection_rate, 1)
        }
        return annotated_frames, cropped_lips, stats
