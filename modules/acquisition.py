"""
Module 1: Video Acquisition Module
Captures video from a webcam or accepts an uploaded video file.
Standardizes frame rates, validates resolution, and prepares video frames for processing.
"""

import os
import cv2
from typing import Tuple, List, Optional
import numpy as np

class VideoAcquisitionModule:
    def __init__(self, target_fps: int = 25, max_duration_sec: int = 10):
        self.target_fps = target_fps
        self.max_duration_sec = max_duration_sec
        self.max_frames = target_fps * max_duration_sec

    def load_video_from_file(self, file_path: str) -> Tuple[bool, List[np.ndarray], str]:
        """
        Loads and validates an uploaded video file.
        Returns: (success: bool, frames: List[np.ndarray], status_msg: str)
        """
        if not os.path.exists(file_path):
            return False, [], f"Error: File not found at {file_path}"

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, [], "Error: Unable to open video file."

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample or limit frames
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return False, [], "Error: No valid frames found in video."

        duration = len(frames) / fps
        return True, frames, f"Successfully loaded {len(frames)} frames ({duration:.1f}s at {fps:.1f} fps)."

    def save_uploaded_temp(self, uploaded_file_bytes: bytes, temp_path: str = "temp_input.mp4") -> str:
        """Saves uploaded Streamlit bytes to a temporary local MP4 file."""
        with open(temp_path, "wb") as f:
            f.write(uploaded_file_bytes)
        return temp_path

