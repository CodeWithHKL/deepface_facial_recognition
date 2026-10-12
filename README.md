# DeepFace facial recognition

Real-time webcam face recognition using [DeepFace](https://github.com/serengil/deepface) (Facenet embeddings) and OpenCV.

## Setup

Use Python 3.9–3.11 (required by `tensorflow==2.15.0`).

```
pip install -r requirements.txt
```

## Dataset

One folder per person, containing clear photos of that person:

```
dataset/
  alice/ 1.jpg 2.jpg
  bob/   1.png
```

If a photo has several faces, the largest one is used. The `dataset/` folder and the embedding cache hold biometric data and are git-ignored.

## Run

```
python main.py
```

Press `q` to quit. Embeddings are cached in `embeddings.npz` and rebuilt automatically when the dataset, model or detector changes.

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `dataset/` | dataset folder |
| `--threshold` | `0.60` | minimum cosine similarity for a match (raise to be stricter) |
| `--model` | `Facenet` | DeepFace model |
| `--detector` | `opencv` | DeepFace detector backend (e.g. `retinaface`, `yunet`) |
| `--every` | `5` | run recognition every N frames |
| `--camera` | `0` | webcam index |
| `--rebuild` | off | force re-enrollment |
