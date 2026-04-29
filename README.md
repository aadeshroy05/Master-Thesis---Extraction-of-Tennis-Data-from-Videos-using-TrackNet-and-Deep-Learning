\# Master Thesis — Extraction of Tennis Data from Videos using TrackNet and Deep Learning



\## Overview

This project extracts structured tennis data (ball trajectory, bounce detection, court detection) from match videos using TrackNet and CatBoost-based classifiers.



\---



\## Project Structure

repo/

├── src/                  # All Python source code

├── input/                # Place your input videos here

├── output/               # Results will be saved here

├── models/               # Place downloaded weights here

├── requirements.txt

└── README.md



\---



\## Installation



\### 1. Clone the repository

```bash

git clone https://github.com/aadeshroy05/Master-Thesis---Extraction-of-Tennis-Data-from-Videos-using-TrackNet-and-Deep-Learning.git

cd Master-Thesis---Extraction-of-Tennis-Data-from-Videos-using-TrackNet-and-Deep-Learning

```



\### 2. Install dependencies

```bash

pip install -r requirements.txt

```



\### 3. Download Model Weights

Download the model weights and place them inside a `models/` folder in the repo:



| Model | Download |

|---|---|

| TrackNet | \[tracknet\_model.pt](https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/tracknet\_model.pt) |

| Court Detection | \[court\_model.pt](https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/court\_model.pt) |

| Bounce Detection | \[bounce\_model.cbm](https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/bounce\_model.cbm) |



Or download via terminal:

```bash

mkdir models

curl -L -o models/tracknet\_model.pt https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/tracknet\_model.pt

curl -L -o models/court\_model.pt https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/court\_model.pt

curl -L -o models/bounce\_model.cbm https://huggingface.co/aadeshroy05/tennis-tracknet-weights/resolve/main/bounce\_model.cbm

```



\---



\## Usage



1\. Place your input video in the `input/` folder

2\. Run the pipeline:

```bash

python src/main.py --video input/your\_match.mp4 --output output/

```

3\. Results will be saved in the `output/` folder



\---



\## Requirements

\- Python 3.8+

\- PyTorch

\- OpenCV

\- CatBoost

\- NumPy



\---



\## Author

Aadesh Roy — Master Thesis, 2025

