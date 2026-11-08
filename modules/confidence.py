"""
Modules 7 & 8: Confidence Evaluation & Feedback Modules
- Calculates and classifies prediction reliability into High, Medium, or Low categories.
- Generates actionable user feedback (e.g., lighting, angle, articulation recommendations).
"""

from typing import Dict, Any

class ConfidenceAndFeedbackModule:
    def __init__(self, high_threshold: float = 80.0, medium_threshold: float = 65.0):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def evaluate_and_provide_feedback(self, confidence: float, tracking_coverage: float) -> Dict[str, Any]:
        """
        Classifies prediction confidence and generates practical feedback.
        
        confidence: Prediction confidence score (0.0 - 100.0)
        tracking_coverage: MediaPipe facial landmark tracking percentage (0.0 - 100.0)
        """
        # Composite score taking into account tracking stability
        effective_score = round((confidence * 0.75) + (tracking_coverage * 0.25), 1)

        if effective_score >= self.high_threshold:
            category = "High Confidence"
            badge_color = "#28a745"  # Green
            feedback_message = (
                "Excellent visual clarity and lip articulation. "
                "The mouth region was tracked continuously with optimal illumination. "
                "The predicted phrase has a very high reliability rating."
            )
            actionable_tip = "Visual conditions are ideal for accurate speech inference."

        elif effective_score >= self.medium_threshold:
            category = "Medium Confidence"
            badge_color = "#ffc107"  # Amber/Yellow
            feedback_message = (
                "Moderate prediction confidence. The system detected speech movement, "
                "but slight visual ambiguity or minor motion blur was observed across some frames."
            )
            actionable_tip = "Try speaking at a slightly steadier pace and ensure your face directly faces the camera."

        else:
            category = "Low Confidence"
            badge_color = "#dc3545"  # Red
            feedback_message = (
                "Low confidence detection. The mouth region experienced significant occlusion, "
                "insufficient lighting, or rapid head movement."
            )
            actionable_tip = "Please improve lighting on your mouth, eliminate head turns, and speak clearly."

        return {
            "raw_confidence": confidence,
            "tracking_coverage": tracking_coverage,
            "effective_score": effective_score,
            "category": category,
            "badge_color": badge_color,
            "feedback": feedback_message,
            "tip": actionable_tip
        }

