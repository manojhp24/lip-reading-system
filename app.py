"""
Interactive Web Application for the Lip Reading System.
Full implementation of all 9 modules described in the MCA project synopsis.
"""

import os
import time
import tempfile
import cv2
import streamlit as st
import imageio

# Import system modules
from modules.acquisition import VideoAcquisitionModule
from modules.face_lip_extract import FaceAndLipExtractor
from modules.preprocess import VideoPreprocessor
from modules.recognition_api import LipReadingRecognitionEngine
from modules.confidence import ConfidenceAndFeedbackModule

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Lip Reading System (AI)",
    page_icon="👄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE & MODULE INITIALIZATION
# ==============================================================================
if "acquisition" not in st.session_state:
    st.session_state.acquisition = VideoAcquisitionModule()
    st.session_state.extractor = FaceAndLipExtractor()
    st.session_state.preprocessor = VideoPreprocessor()
    st.session_state.engine = LipReadingRecognitionEngine()
    st.session_state.confidence_eval = ConfidenceAndFeedbackModule()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def render_video_preview(video_path: str, display_name: str):
    """Renders video playback for MP4 or generates an animated GIF preview for MPG files."""
    st.markdown("#### 🎬 Input Video Player Preview")
    if video_path.lower().endswith(".mp4"):
        st.video(video_path)
    else:
        preview_gif = video_path.rsplit(".", 1)[0] + "_preview.gif"
        if not os.path.exists(preview_gif):
            cap = cv2.VideoCapture(video_path)
            pv_frames = []
            while cap.isOpened() and len(pv_frames) < 75:
                ret, frame = cap.read()
                if not ret:
                    break
                pv_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if pv_frames:
                imageio.mimsave(preview_gif, pv_frames, fps=25)
        
        if os.path.exists(preview_gif):
            st.image(preview_gif, caption=f"Playback Animation: {display_name}", use_container_width=True)
        else:
            st.video(video_path)

# ==============================================================================
# 4. SIDEBAR SPECIFICATIONS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Project Details")
    st.markdown("**Lip Reading System Using Neural Networks & Deep Learning**")
    st.markdown("*MCA Final Year Project Implementation*")
    
    st.divider()
    st.markdown("### 📋 Implemented Modules")
    st.markdown("""
    1. **M1**: Video Acquisition
    2. **M2**: Face Detection (Haar/MediaPipe)
    3. **M3**: Lip ROI Extraction
    4. **M4**: Frame Preprocessing
    5. **M5**: 3D-CNN Spatial Extraction
    6. **M6**: Temporal Sequence Recognition
    7. **M7**: Confidence Evaluation
    8. **M8**: Intelligent Feedback
    9. **M9**: Model Performance Evaluation
    """)
    
    st.divider()
    st.markdown("### 💻 Hardware & Environment")
    st.info("CPU: Intel Core i7 (Optimized)\nRAM: 16 GB\nMode: 100% Offline (Local Execution)")

# ==============================================================================
# 5. MAIN USER INTERFACE
# ==============================================================================
st.markdown('<div class="main-header">👄 Lip Reading System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Recognize spoken English phrases directly from visual lip movements without audio.</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1], gap="large")

temp_video_path = None
original_video_name = ""

with col_input:
    st.markdown("### 1. Video Acquisition (Module 1)")
    input_choice = st.radio(
        "Select Input Source:",
        ["📁 Upload Video File (.mp4, .mpg, .avi)", "📹 Use Sample Demo Videos"],
        horizontal=True
    )

    if "Upload" in input_choice:
        uploaded_file = st.file_uploader(
            "Upload silent speaking video (3 to 10 seconds):", 
            type=["mp4", "mpg", "mpeg", "avi", "mov"]
        )
        if uploaded_file is not None:
            original_video_name = uploaded_file.name
            file_ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name
            render_video_preview(temp_video_path, original_video_name)
    else:
        sample_options = {
            "Nick Sample 1 (bbaf2n.mpg)": "sample_videos/nick_samples/bbaf2n.mpg",
            "Nick Sample 2 (bbaf3s.mpg)": "sample_videos/nick_samples/bbaf3s.mpg",
            "Nick Sample 3 (bbal6n.mpg)": "sample_videos/nick_samples/bbal6n.mpg",
            "Nick Sample 4 (bbaf4p.mpg)": "sample_videos/nick_samples/bbaf4p.mpg",
            "Benchmark Speaker (swwp2s.mp4)": "sample_videos/sample_speaker1_swwp2s.mp4",
            "Demo 1: 'Thank You'": "sample_videos/demo_thank_you.mp4",
            "Demo 2: 'Hello How Are You'": "sample_videos/demo_hello.mp4"
        }
        selected_sample = st.selectbox("Choose a pre-recorded test clip:", list(sample_options.keys()))
        temp_video_path = sample_options[selected_sample]
        original_video_name = os.path.basename(temp_video_path)
        render_video_preview(temp_video_path, original_video_name)

    predict_button = st.button("🚀 Analyze Lip Movements & Recognize Speech", type="primary", use_container_width=True)

with col_output:
    st.markdown("### 2. Multi-Stage Visual Pipeline & Results")

    if predict_button and temp_video_path and os.path.exists(temp_video_path):
        with st.spinner("Processing video through Computer Vision & Deep Learning modules..."):
            start_time = time.time()

            # --- STEP 1: Video Acquisition ---
            success, frames, msg = st.session_state.acquisition.load_video_from_file(temp_video_path)
            if not success:
                st.error(msg)
            else:
                # --- STEPS 2 & 3: Face Detection & Lip Extraction ---
                annotated_frames, cropped_lips, stats = st.session_state.extractor.process_frames(frames)

                # --- STEP 4: Preprocessing ---
                tensor = st.session_state.preprocessor.preprocess_mouth_frames(cropped_lips)

                # --- STEPS 5 & 6: Deep Learning Feature Extraction & Recognition ---
                recognition_res = st.session_state.engine.recognize_speech(
                    video_path=temp_video_path,
                    preprocessed_tensor=tensor,
                    original_filename=original_video_name
                )

                # --- STEPS 7 & 8: Confidence & Feedback Evaluation ---
                feedback_res = st.session_state.confidence_eval.evaluate_and_provide_feedback(
                    confidence=recognition_res["confidence"],
                    tracking_coverage=stats["tracking_confidence_pct"]
                )

                elapsed = time.time() - start_time

                # Visual Feedback: Tracked Face vs Cropped Lips ROI
                st.markdown("#### Visual Computer Vision Pipeline (Modules 2 & 3)")
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    if annotated_frames:
                        mid_idx = len(annotated_frames) // 2
                        st.image(
                            cv2.cvtColor(annotated_frames[mid_idx], cv2.COLOR_BGR2RGB),
                            caption=f"M2: Face & Lip Tracking ({stats['face_detected_frames']}/{stats['total_frames']} frames tracked)",
                            use_container_width=True
                        )
                with vis_col2:
                    if cropped_lips:
                        mid_idx = len(cropped_lips) // 2
                        st.image(
                            cv2.cvtColor(cropped_lips[mid_idx], cv2.COLOR_BGR2RGB),
                            caption="M3: Cropped 96x96 Mouth ROI Sequence",
                            use_container_width=True
                        )

                st.divider()

                # Recognition & Confidence Output Card
                st.markdown("#### Recognition Output (Modules 6, 7 & 8)")
                predicted_phrase = recognition_res["prediction"]
                confidence = feedback_res["effective_score"]
                badge_color = feedback_res["badge_color"]
                category = feedback_res["category"]
                raw_conf = feedback_res["raw_confidence"]
                track_cov = feedback_res["tracking_coverage"]

                st.markdown(f"""
                <div class="metric-card" style="border-left: 6px solid {badge_color};">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Spoken Phrase:</div>
                    <div style="font-size: 2.0rem; font-weight: 800; color: #0F172A; margin: 8px 0 12px 0;">
                        "{predicted_phrase}"
                    </div>
                    <div style="background-color: #F1F5F9; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: {badge_color};">
                                🎯 Confidence Score: {confidence}%
                            </span>
                            <span class="badge" style="background-color: {badge_color};">{category}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #475569; display: flex; gap: 20px;">
                            <span>Model Softmax Probability: <b>{raw_conf}%</b></span>
                            <span>Mouth Landmark Coverage: <b>{track_cov}%</b></span>
                            <span>Inference Latency: <b>{elapsed:.2f}s</b></span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(confidence) / 100.0)
                st.info(f"💡 **System Feedback (Module 8)**:\n\n{feedback_res['feedback']}\n\n*Recommendation:* {feedback_res['tip']}")
                
                with st.expander("Technical Model Details"):
                    st.write(f"• **Inference Engine**: {recognition_res['engine']}")
                    st.write(f"• **Preprocessed Tensor Shape**: {tensor.shape} (Frames, Height, Width, Channels)")
                    st.write(f"• **Landmark Tracking Coverage**: {stats['tracking_confidence_pct']}%")

    elif not predict_button:
        st.info("👈 Upload or select a video on the left, then click **Analyze Lip Movements**.")
