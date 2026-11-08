"""
Module 4: Preprocessing Module
Resizes, normalizes, grayscales, aligns, and standardizes frame sequences for deep learning models.
"""

import cv2
import numpy as np
from typing import List, Tuple

class VideoPreprocessor:
    def __init__(self, target_shape: Tuple[int, int] = (96, 96), max_sequence_len: int = 75):
        """
        target_shape: (Width, Height)
        max_sequence_len: Fixed temporal length (frames) expected by standard lip-reading sequence models.
        """
        self.target_shape = target_shape
        self.max_sequence_len = max_sequence_len

    def preprocess_mouth_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Applies standard deep learning visual speech preprocessing:
        1. Convert BGR to Grayscale.
        2. Resize to exact target dimension.
        3. Normalize pixel intensities (Zero mean & unit variance: (I - mean) / std).
        4. Pad or truncate to fixed temporal length.

        Returns:
            processed_tensor: NumPy array of shape (Sequence_Len, Height, Width, 1)
        """
        processed_frames = []

        for f in frames:
            # Grayscale conversion
            if len(f.shape) == 3 and f.shape[2] == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                gray = f

            # Resize
            resized = cv2.resize(gray, self.target_shape)

            # Normalization (Z-score standardisation)
            mean = np.mean(resized)
            std = np.std(resized) + 1e-6
            norm = (resized - mean) / std
            processed_frames.append(norm)

        # Temporal alignment (Padding or Truncating)
        curr_len = len(processed_frames)
        if curr_len < self.max_sequence_len:
            # Pad with zero frames
            pad_amount = self.max_sequence_len - curr_len
            for _ in range(pad_amount):
                processed_frames.append(np.zeros(self.target_shape, dtype=np.float32))
        else:
            # Truncate
            processed_frames = processed_frames[:self.max_sequence_len]

        tensor = np.array(processed_frames, dtype=np.float32)
        # Add channel dimension: (T, H, W, 1)
        tensor = np.expand_dims(tensor, axis=-1)
        return tensor

    def save_preview_video(self, frames: List[np.ndarray], output_path: str, fps: int = 25) -> str:
        """Saves a processed preview video to disk for UI visualization."""
        if not frames:
            return ""
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        out.release()
        return output_path

