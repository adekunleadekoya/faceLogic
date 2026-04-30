# FaceLogic — Face Matching App

A desktop application that finds the closest matching face(s) from a local image library using deep learning embeddings.

## How it works

1. On startup, the app scans the `pics/` folder and builds a face embedding index using [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l` model).
2. The user selects a query image via the GUI.
3. The app extracts the face embedding from the query image and searches the index using [FAISS](https://github.com/facebookresearch/faiss) (cosine similarity).
4. The top matching images are displayed as a scrollable card grid with similarity scores.

## Tech stack

| Component | Library |
|-----------|---------|
| Face detection & recognition | InsightFace `buffalo_l` (SCRFD + ArcFace) |
| Similarity search | FAISS `IndexFlatIP` (inner-product / cosine) |
| Image decoding | OpenCV |
| GUI | Python Tkinter |
| Image display | Pillow |

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

> **Note:** InsightFace will automatically download the `buffalo_l` model (~300 MB) on the first run.

## Usage

```bash
python app.py
```

1. Wait for the model and index to load (progress shown in the status bar).
2. Click **Browse** to select a query image.
3. Click **Find Match** to search.
4. Matched faces are shown as image cards ranked by similarity score.

## Folder structure

```
face-match-app/
├── app.py            # Main application
├── requirements.txt  # Python dependencies
├── pics/             # Image library to search against
└── README.md
```

## Notes

- Only the **first detected face** in each image is indexed.
- Images larger than 640 px are automatically downscaled before detection.
- The `buffalo_l` model runs on CPU by default; GPU (CUDA) can be enabled by changing the ONNX provider in `load_model()`.
