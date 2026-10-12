"""Real-time face recognition using DeepFace embeddings and a webcam."""
import argparse
import hashlib
import os
import sys

import cv2
import numpy as np
from deepface import DeepFace

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DETECT_WIDTH = 640  # frames are downscaled to this width before detection


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dataset/", help="folder with one subfolder of images per person")
    p.add_argument("--cache", default="embeddings.npz", help="where enrolled embeddings are cached")
    p.add_argument("--rebuild", action="store_true", help="ignore the cache and re-enroll the dataset")
    p.add_argument("--model", default="Facenet", help="DeepFace recognition model")
    p.add_argument("--detector", default="opencv", help="DeepFace detector backend")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="minimum cosine similarity to accept a match (Facenet cosine distance 0.40 = 0.60)")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--every", type=int, default=5, help="run recognition every N frames")
    return p.parse_args()


def normalize(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    return v / np.linalg.norm(v)


def dataset_images(dataset_path):
    """Yield (person, path) for every image in dataset_path/<person>/."""
    for person in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, person)
        if not os.path.isdir(folder):
            continue
        for file in sorted(os.listdir(folder)):
            if file.lower().endswith(IMAGE_EXTENSIONS):
                yield person, os.path.join(folder, file)


def dataset_signature(dataset_path, model, detector):
    """Hash of the settings and image files, used to detect a stale cache."""
    h = hashlib.sha256(f"{model}|{detector}".encode())
    for _, path in dataset_images(dataset_path):
        st = os.stat(path)
        h.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def enroll(dataset_path, model, detector):
    """Embed every dataset image. Returns (names, matrix of normalized embeddings)."""
    names, vectors = [], []
    for person, path in dataset_images(dataset_path):
        try:
            faces = DeepFace.represent(path, model_name=model, detector_backend=detector,
                                       enforce_detection=True)
        except Exception as e:
            print(f"Skipped {path}: {e}")
            continue
        # A photo may contain several faces; assume the largest one is the person.
        face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        names.append(person)
        vectors.append(normalize(face["embedding"]))
        print(f"Enrolled {os.path.basename(path)} for {person}")
    if not vectors:
        return np.array([], dtype=str), np.empty((0, 0), dtype=np.float32)
    return np.array(names), np.stack(vectors)


def load_known_faces(args):
    """Load enrolled embeddings from the cache, re-enrolling if it is missing or stale."""
    signature = dataset_signature(args.dataset, args.model, args.detector)
    if not args.rebuild and os.path.exists(args.cache):
        with np.load(args.cache, allow_pickle=False) as data:
            if str(data["signature"]) == signature:
                print(f"Loaded cached embeddings from {args.cache}")
                return data["names"], data["embeddings"]
    print("Enrolling faces from dataset...")
    names, embeddings = enroll(args.dataset, args.model, args.detector)
    if len(names):
        np.savez(args.cache, names=names, embeddings=embeddings, signature=signature)
    return names, embeddings


def identify(embedding, names, embeddings, threshold):
    """Return (name, similarity) of the best match, or ("Unknown", similarity)."""
    sims = embeddings @ normalize(embedding)
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return "Unknown", float(sims[best])
    return str(names[best]), float(sims[best])


def recognize(frame, args, names, embeddings):
    """Detect and identify all faces in a frame. Returns [(x, y, w, h, label)] in frame coordinates."""
    scale = min(1.0, DETECT_WIDTH / frame.shape[1])
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
    try:
        faces = DeepFace.represent(small, model_name=args.model, detector_backend=args.detector,
                                   enforce_detection=True)
    except ValueError:  # no face found
        return []
    results = []
    for face in faces:
        a = face["facial_area"]
        name, sim = identify(face["embedding"], names, embeddings, args.threshold)
        results.append((int(a["x"] / scale), int(a["y"] / scale),
                        int(a["w"] / scale), int(a["h"] / scale), f"{name} ({sim:.2f})"))
    return results


def main():
    args = parse_args()
    if not os.path.isdir(args.dataset):
        sys.exit(f"Dataset folder not found: {args.dataset} (expected {args.dataset}<person>/<image>.jpg)")

    names, embeddings = load_known_faces(args)
    if not len(names):
        sys.exit("No faces could be enrolled from the dataset.")
    print(f"Known people: {', '.join(sorted(set(names.tolist())))}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    print("Starting webcam... Press 'q' to quit.")

    results, frame_count = [], 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % args.every == 0:
                results = recognize(frame, args, names, embeddings)
            frame_count += 1

            for x, y, w, h, label in results:
                color = (0, 255, 0) if not label.startswith("Unknown") else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Real-Time Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
