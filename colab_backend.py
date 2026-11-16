# -*- coding: utf-8 -*-
"""
Auto-AVSR Free Cloud GPU Backend (Google Colab)
================================================
This script runs the State-of-the-Art Auto-AVSR Visual Speech Recognition model 
on Google Colab's FREE NVIDIA T4 GPU and provides an API link for your local app!

INSTRUCTIONS:
1. Open https://colab.research.google.com and click "New Notebook".
2. Set Runtime to GPU: Menu -> Runtime -> Change runtime type -> Select T4 GPU -> Save.
3. Paste all this code into a cell and press Run (Play button).
4. After 2 minutes, you will see a public Gradio URL (e.g. https://xxxx.gradio.live).
5. Copy that URL and paste it into your local app!
"""

# Step 1: Install Auto-AVSR dependencies
!pip install -q gradio torch torchvision torchaudio sentencepiece mediapipe opencv-python hydra-core

# Step 2: Clone Auto-AVSR repository
!git clone -q https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages.git
%cd Visual_Speech_Recognition_for_Multiple_Languages

# Step 3: Download pretrained LRS3 Lip-Reading weights (English VSR)
!mkdir -p benchmarks/lrs3/models
!wget -q -c https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/releases/download/v1.0/LRS3_V_WER34.7.pth -O benchmarks/lrs3/models/LRS3_V_WER34.7.pth

# Step 4: Define the VSR inference function
import torch
import gradio as gr
from pipelines.pipeline import InferencePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Auto-AVSR Pipeline on {device}...")

pipeline = InferencePipeline(
    config_filename="configs/LRS3_V_WER34.7.ini",
    detector="mediapipe",
    face_track=True,
    device=device
)

def predict_silent_video(video_path):
    """Predicts words/sentences spoken in any silent video."""
    try:
        transcript = pipeline(data_filename=video_path)
        if not transcript or len(transcript.strip()) == 0:
            transcript = "Could not clearly identify lip movements. Please speak clearly facing the camera."
        return transcript.strip().upper()
    except Exception as e:
        return f"Recognition Error: {str(e)}"

# Step 5: Launch Gradio API with public URL
print("\n" + "="*50)
print("Auto-AVSR Cloud API is launching! Copy the public link below:")
print("="*50 + "\n")

demo = gr.Interface(
    fn=predict_silent_video,
    inputs=gr.Video(label="Input Silent Video"),
    outputs=gr.Textbox(label="Predicted Spoken Text"),
    title="Auto-AVSR Lip Reading API"
)

demo.launch(share=True, debug=False)

