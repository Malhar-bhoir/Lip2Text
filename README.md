# Lip2Text


# 🧠 Lip2Text: Lip Reading with LipNet + Streamlit

This project is a lightweight lipreading demo built on [LipNet](https://github.com/oxford-cs/LipNet) and powered by TensorFlow and Streamlit. It takes silent video input of a speaker and predicts the spoken sentence using a trained deep learning model.

---

## 🚀 Features

- ✅ LipNet-based video-to-text inference
- 🎥 FFmpeg-powered video preprocessing
- 🖼️ GIF animation preview of cropped frames
- 📦 Streamlit UI for easy interaction
- 🧪 Supports unseen `.mpg` videos for testing

---

## 📁 Folder Structure

```
LipNet-main/
├── app/
│   └── streamlitapp.py         # Main Streamlit interface
├── utils/
│   └── util.py                 # Video + alignment loader
│   └── modelutil.py            # Model architecture + weights
├── models/
│   └── checkpoint              # Pretrained weights
├── data/
│   ├── s1/                     # Input videos (.mpg)
│   └── alignments/s1/         # Optional .align files
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions (Windows)

### 1. Install Python 3.9.23
Download from [python.org](https://www.python.org/downloads/release/python-3923/)  
Choose: `Windows installer (64-bit)`

### 2. Install FFmpeg
Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)  
Extract and add `ffmpeg/bin` to your system PATH

### 3. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ Use the compatible `requirements.txt`:
```txt
tensorflow==2.10.1
streamlit==1.18.0
opencv-python==4.8.0.76
imageio==2.31.1
protobuf==3.19.6
```

---

## 📹 How to Add New Videos

1. Convert your video to `.mpg` format:
   ```bash
   ffmpeg -i input.mp4 -vf scale=360:288 -qscale:v 2 output.mpg
   ```

2. Ensure it has ~75 frames (≈3 sec at 25 fps)

3. Place it in:
   ```
   LipNet-main/data/s1/
   ```

4. (Optional) Add a matching `.align` file in:
   ```
   LipNet-main/data/alignments/s1/
   ```

---

## 🧪 Run the App

```bash
streamlit run app/streamlitapp.py
```

---



---

## 🙌 Acknowledgements

- [LipNet (Oxford)](https://github.com/oxford-cs/LipNet)
- [Streamlit](https://streamlit.io/)
- [FFmpeg](https://ffmpeg.org/)
```

---


