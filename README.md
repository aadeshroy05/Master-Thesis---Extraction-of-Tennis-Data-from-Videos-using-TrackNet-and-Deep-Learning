# Master Thesis — Extraction of Tennis Data from Videos using TrackNet and Deep Learning

## Overview
This project extracts structured tennis data (ball trajectory, bounce detection, court detection) from match videos using TrackNet and CatBoost-based classifiers.

---

## Project Structure
```
repo/
├── src/                  # All Python source code
├── input/                # Place your input videos here
├── output/               # Results will be saved here
├── models/               # Place downloaded weights here
├── requirements.txt
└── README.md
```
---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/aadeshroy05/Master-Thesis---Extraction-of-Tennis-Data-from-Videos-using-TrackNet-and-Deep-Learning.git
cd Master-Thesis---Extraction-of-Tennis-Data-from-Videos-using-TrackNet-and-Deep-Learning
```

### 2. Create required folders
```bash
mkdir models input output
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
Download the model weights and place them inside the `models/` folder:

| Model | Download |
|---|---|
| TrackNet | [tracknet_model.pt](https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/tracknet_model.pt) |
| Court Detection | [court_model.pt](https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/court_model.pt) |
| Bounce Detection | [bounce_model.cbm](https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/bounce_model.cbm) |

Or download via terminal:
```bash
curl -L -o models/tracknet_model.pt https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/tracknet_model.pt
curl -L -o models/court_model.pt https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/court_model.pt
curl -L -o models/bounce_model.cbm https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/bounce_model.cbm
```

---

## Usage

### ⚠️ Video Resolution Requirement
Input video **must be 1280x720 (720p) resolution minimum**.
Lower resolution videos (e.g. 640x360) will cause court detection to fail.

If your video is lower resolution, upscale it first:
```bash
python -c "
import cv2
cap = cv2.VideoCapture('input/your_video.mp4')
out = cv2.VideoWriter('input/your_video_upscaled.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    out.write(frame)
cap.release()
out.release()
print('Done!')
"
```

### Run the pipeline
1. Place your input video in the `input/` folder
2. Run:
```bash
python src/main.py --path_ball_track_model models/tracknet_model.pt --path_court_model models/court_model.pt --path_bounce_model models/bounce_model.cbm --path_input_video input/your_match.mp4 --path_output_video output/result.mp4
```
3. Results (annotated video + `bounces.xml`) will be saved in the `output/` folder

---

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- CatBoost
- NumPy
- EasyOCR
- SceneDetect

---

## Author
Aadesh Roy — Master Thesis, 2026
