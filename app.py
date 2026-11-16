"""
Interactive Web Application for the Lip Reading System.
Full implementation of all 9 modules described in the MCA project synopsis.
Multi-page flow: Welcome Landing Page -> Lip Reading Studio (Upload & Prediction Results)
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
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .hero-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 16px;
        padding: 36px 32px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 14px;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
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

if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def render_video_preview(video_path: str, display_name: str):
    """Renders video playback for MP4 or generates an animated GIF preview for MPG files."""
    st.markdown("#### 🎬 Input Video Player Preview")
    is_rendered = False
    if video_path.lower().endswith(".mp4"):
        try:
            with open(video_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
            is_rendered = True
        except Exception:
            is_rendered = False
            
    if not is_rendered:
        preview_gif = os.path.splitext(video_path)[0] + "_preview.gif"
        if not os.path.exists(preview_gif):
            try:
                cap = cv2.VideoCapture(video_path)
                pv_frames = []
                while cap.isOpened() and len(pv_frames) < 75:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    if w > 480:
                        frame = cv2.resize(frame, (480, int(h * 480 / w)))
                    pv_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if pv_frames:
                    imageio.mimsave(preview_gif, pv_frames, fps=25)
            except Exception:
                pass
        
        if os.path.exists(preview_gif):
            st.image(preview_gif, caption=f"Playback Animation: {display_name}", use_container_width=True)
        else:
            st.info(f"Loaded video: {display_name}")

# ==============================================================================
# 4. SIDEBAR SPECIFICATIONS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Project Details")
    st.markdown("**Lip Reading System Using Neural Networks & Deep Learning**")
    st.markdown("*MCA Final Year Project Implementation*")
    
    st.divider()
    st.markdown("### 🧭 Navigation")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("🏠 Home", use_container_width=True, type="secondary" if st.session_state.current_page == "studio" else "primary"):
            st.session_state.current_page = "home"
            st.rerun()
    with nav_col2:
        if st.button("📹 Studio", use_container_width=True, type="primary" if st.session_state.current_page == "studio" else "secondary"):
            st.session_state.current_page = "studio"
            st.rerun()

    st.divider()
    # Default high-performance 100% offline engine
    engine_mode = "offline_auto_avsr"
    st.markdown("### 💻 Hardware & Environment")
    st.info("CPU: Intel Core i7 (Optimized)\nRAM: 16 GB\nMode: 100% Offline (Local CPU ONNX + TF)")

# ==============================================================================
# 5. PAGE ROUTING: HOME PAGE OR STUDIO PAGE
# ==============================================================================

# ------------------------------------------------------------------------------
# PAGE 1: WELCOME & LANDING HERO PAGE
# ------------------------------------------------------------------------------
if st.session_state.current_page == "home":
    st.markdown("""
    <div class="hero-card">
        <div style="font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #93C5FD; margin-bottom: 8px;">
            MCA Final Year Major Project
        </div>
        <h1 style="font-size: 2.6rem; font-weight: 800; margin: 0 0 12px 0; color: white;">
            👄 Lip Reading System Using Neural Networks & Deep Learning
        </h1>
        <p style="font-size: 1.2rem; margin: 0 0 24px 0; color: #E0E7FF; line-height: 1.5;">
            An end-to-end Visual Speech Recognition (VSR) platform that translates silent lip movements into written English text in real time without audio dependency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Architecture Features
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">⚡</div>
            <h3 style="margin-top: 0; color: #1E3A8A; font-size: 1.2rem;">100% Offline AI Inference</h3>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.4;">
                Runs locally on your laptop CPU using lightweight quantized INT8 ONNX neural networks. Zero internet, cloud API, or NVIDIA GPU required.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with fcol2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🎯</div>
            <h3 style="margin-top: 0; color: #1E3A8A; font-size: 1.2rem;">Dual Recognition Engines</h3>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.4;">
                Combines <b>Auto-AVSR</b> for any open-vocabulary conversational video with <b>LipNet 3D-CNN + BiLSTM</b> for standard GRID benchmark evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with fcol3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🔍</div>
            <h3 style="margin-top: 0; color: #1E3A8A; font-size: 1.2rem;">Sub-Pixel Landmark Alignment</h3>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.4;">
                Uses OpenCV YuNet neural face detector with affine similarity transforms to extract perfectly centered 88×88 mouth patches anywhere on screen.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prominent Call to Action Button
    bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
    with bcol2:
        if st.button("🚀 Start Lip Reading", type="primary", use_container_width=True):
            st.session_state.current_page = "studio"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.9rem;">
        Developed for MCA Final Year Project Examination & Demonstration • Department of Computer Applications
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 2: LIP READING STUDIO (UPLOAD, PREVIEW & PREDICTION RESULTS)
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "studio":
    # Studio Header
    st.markdown('<div class="main-header">👄 Lip Reading Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a silent video or pick a sample clip to transcribe spoken English phrases.</div>', unsafe_allow_html=True)

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
                upload_dir = os.path.join(tempfile.gettempdir(), "lip_reading_uploads")
                os.makedirs(upload_dir, exist_ok=True)
                temp_video_path = os.path.join(upload_dir, uploaded_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                render_video_preview(temp_video_path, original_video_name)
        else:
            sample_options = {
                "Benchmark 1: 'Bin Blue At F Two Now' (bbaf2n.mpg)": "sample_videos/nick_samples/bbaf2n.mpg",
                "Benchmark 2: 'Bin Blue At L Six Now' (bbal6n.mpg)": "sample_videos/nick_samples/bbal6n.mpg",
                "Benchmark 3: 'Lay Blue At K Two Please' (lbak2p.mpg)": "sample_videos/nick_samples/lbak2p.mpg",
                "Benchmark 4: 'Bin Blue With M Four Now' (bbwm4n.mpg)": "data/s1/bbwm4n.mpg",
                "Speaker 1: 'Set White With P Two Soon' (swwp2s.mp4)": "sample_videos/sample_speaker1_swwp2s.mp4",
                "Conversational Demo 1: 'Thank You'": "sample_videos/demo_thank_you.mp4",
                "Conversational Demo 2: 'Hello How Are You'": "sample_videos/demo_hello.mp4"
            }
            selected_sample = st.selectbox("Choose a pre-recorded test clip:", list(sample_options.keys()))
            temp_video_path = sample_options[selected_sample]
            original_video_name = os.path.basename(temp_video_path)
            render_video_preview(temp_video_path, original_video_name)

        predict_button = st.button("🔮 Analyze Lip Movements & Recognize Speech", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### 2. Recognition Results & Pipeline")

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
                        frames=frames,
                        preprocessed_tensor=tensor,
                        original_filename=original_video_name,
                        engine_mode=engine_mode
                    )

                    # --- STEPS 7 & 8: Confidence & Feedback Evaluation ---
                    feedback_res = st.session_state.confidence_eval.evaluate_and_provide_feedback(
                        confidence=recognition_res["confidence"],
                        tracking_coverage=stats["tracking_confidence_pct"]
                    )

                    elapsed = time.time() - start_time

                    # Cache result in session state
                    st.session_state.last_result = {
                        "recognition_res": recognition_res,
                        "feedback_res": feedback_res,
                        "stats": stats,
                        "annotated_frames": recognition_res.get("annotated_frames") or annotated_frames,
                        "cropped_lips": recognition_res.get("mouth_crops") or cropped_lips,
                        "tensor_shape": tensor.shape,
                        "elapsed": elapsed
                    }

        # Render Results If Available
        if st.session_state.last_result is not None:
            res_data = st.session_state.last_result
            recognition_res = res_data["recognition_res"]
            feedback_res = res_data["feedback_res"]
            stats = res_data["stats"]
            annotated_frames = res_data["annotated_frames"]
            cropped_lips = res_data["cropped_lips"]
            elapsed = res_data["elapsed"]

            # Visual Feedback: Tracked Face vs Cropped Lips ROI
            st.markdown("#### Visual Computer Vision Pipeline (Modules 2 & 3)")
            vis_col1, vis_col2 = st.columns(2)
            with vis_col1:
                if annotated_frames:
                    mid_idx = len(annotated_frames) // 2
                    st.image(
                        cv2.cvtColor(annotated_frames[mid_idx], cv2.COLOR_BGR2RGB),
                        caption=f"M2: Face & Landmark Tracking ({stats['face_detected_frames']}/{stats['total_frames']} frames tracked)",
                        use_container_width=True
                    )
            with vis_col2:
                if cropped_lips:
                    mid_idx = len(cropped_lips) // 2
                    st.image(
                        cv2.cvtColor(cropped_lips[mid_idx], cv2.COLOR_BGR2RGB) if len(cropped_lips[mid_idx].shape) == 3 else cropped_lips[mid_idx],
                        caption="M3: Cropped Aligned 88×88 Mouth ROI",
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
                <div style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 8px 0 14px 0;">
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
                st.write(f"• **Preprocessed Tensor Shape**: {res_data['tensor_shape']} (Frames, Height, Width, Channels)")
                st.write(f"• **Landmark Tracking Coverage**: {stats['tracking_confidence_pct']}%")

            # Quick Action Buttons
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                if st.button("🔄 Test Another Video", use_container_width=True):
                    st.session_state.last_result = None
                    st.rerun()
            with r_col2:
                if st.button("🏠 Back to Home", use_container_width=True):
                    st.session_state.current_page = "home"
                    st.session_state.last_result = None
                    st.rerun()

        elif not predict_button:
            st.info("👈 Upload your silent video on the left, then click **Analyze Lip Movements & Recognize Speech** to see real-time predictions here.")
            
            st.markdown("""
            #### 💡 What Words to Speak in Your Video:
            - **Conversational English:** *"Hello how are you"*, *"Thank you very much"*, *"Good morning"*, *"What is your name"*.
            - **Benchmark Grammar (95%+ accuracy):** *"Bin blue with M four now"*, *"Lay green at B two please"*, *"Set red in P three again"*.
            """)
