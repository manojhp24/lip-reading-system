# 👄 Lip Reading System Using Neural Networks and Deep Learning

**MCA Final Year Project — Visual Speech Recognition (VSR)**

An end-to-end deep learning system that recognizes spoken words or phrases by analyzing visual lip movements from video input without depending on audio. Built using 3D-CNNs, Bidirectional LSTMs, and CTC Decoding on the GRID audio-visual corpus.

---

## 💻 System & Hardware Requirements

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 / 11, Ubuntu 20.04+, or macOS | Windows 10/11 |
| **Python Version** | **Python 3.10** (Recommended: 3.10.x) | Python 3.10.11 |
| **Processor (CPU)** | Intel Core i5 / AMD Ryzen 5 | Intel Core i5/i7 (No dedicated GPU required) |
| **RAM** | 8 GB | 16 GB |
| **Internet** | Not required for execution | **100% Offline Capable** |

> ⚠️ **Important Python Note:** You **must use Python 3.10**. Newer versions (like Python 3.12 or 3.14) have compatibility issues with legacy TensorFlow 2 / MediaPipe C++ bindings.

---

## 🚀 How to Run on Any Computer (Step-by-Step Guide for Friends)

Share the entire project folder (`lip-reading/`) with your friend (e.g. via USB drive, Google Drive, or ZIP).

### Step 1: Install Python 3.10 (If not already installed)
1. Download **Python 3.10.11** from the official website:
   👉 [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/)
2. Run the installer and **make sure to CHECK the box: `Add Python 3.10 to PATH`**.

---

### Step 2: Open Terminal / PowerShell in the Project Folder
Open PowerShell or Command Prompt, navigate to the project directory:

```bash
cd lip-reading
```

---

### Step 3: Create and Activate the Virtual Environment

**On Windows:**
```powershell
# Create virtual environment using Python 3.10
py -3.10 -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

*(You will see `(venv)` appear at the beginning of your terminal prompt).*

---

### Step 4: Install Dependencies

With the virtual environment activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 5: Launch the Application

**Option A (Easiest - 1 Click):**
* Simply double-click the **`run_app.bat`** file inside the project folder.

**Option B (From Terminal):**
```bash
streamlit run app.py
```

The application will automatically launch in your default web browser at:
🌐 **`http://localhost:8501`**

---

## 🎯 How to Test and Present the System

1. In the web interface under **1. Video Acquisition (Module 1)**:
   * Select **"📹 Use Sample Demo Videos"** and pick any sample (e.g., `Nick Sample 1 (bbaf2n.mpg)`).
   * OR select **"📁 Upload Video File"** and upload any `.mpg` or `.mp4` file from `sample_videos/`.
2. View the speaking person in the **Input Video Player Preview**.
3. Click **"🚀 Analyze Lip Movements & Recognize Speech"**.
4. The system will:
   * Detect the speaker's face and track the mouth bounding box.
   * Display the cropped **96×96 mouth sequence**.
   * Run the pre-trained **LipNet 3D-CNN + BiLSTM** neural network.
   * Output the recognized spoken sentence along with the **Confidence Score Card** (Softmax probability, landmark coverage, and latency).

---

## 📁 Project Structure

```
lip-reading/
│
├── app.py                       # Main Streamlit web application
├── run_app.bat                  # One-click Windows launcher
├── requirements.txt             # Exact verified Python dependencies
├── README.md                    # Project documentation & run guide
│
├── modules/                     # Core system modules (MCA Synopsis)
│   ├── acquisition.py           # Module 1: Video Acquisition
│   ├── face_lip_extract.py      # Modules 2 & 3: Face Detection & Lip Cropping
│   ├── preprocess.py            # Module 4: Normalization & Frame Padding
│   ├── recognition_api.py       # Modules 5 & 6: LipNet Deep Learning Engine
│   ├── confidence.py            # Modules 7 & 8: Confidence Score & Smart Feedback
│   ├── evaluation.py            # Module 9: Evaluation Metrics & Confusion Matrix
│   ├── nick_loader.py           # TensorFlow 2 Model Architecture Loader
│   ├── nick_utils.py            # CTC greedy decoder & vocabulary mappings
│   └── haarcascade_frontalface_default.xml # Bundled offline face detector
│
├── models/                      # Pre-trained deep learning weights & alignment index
│   ├── checkpoint
│   ├── checkpoint.data-00000-of-00001
│   ├── checkpoint.index
│   └── all_grid_alignments.txt
│
├── sample_videos/               # Ready-to-demo test videos
│   └── nick_samples/            # Official benchmark test clips (.mpg)
│
└── data/                        # Dataset videos and alignment annotations
```

---

## 🛡️ Key Features for Project Defense & Viva

* **100% Offline Capability**: Runs entirely locally on CPU without requiring an internet connection or external API keys.
* **No GPU Dependency**: Uses Intel oneDNN-optimized `tensorflow-cpu` to execute inferences on standard laptops within ~1 to 2 seconds.
* **Mathematical Confidence Scoring**: Evaluates prediction certainty by combining neural network softmax probabilities with computer vision tracking coverage.
