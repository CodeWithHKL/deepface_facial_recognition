import cv2
import os
import numpy as np
from deepface import DeepFace

# ---------------------------
# CONFIGURATION
# ---------------------------
dataset_path = "dataset/"       # dataset folder with subfolders per person
model_name = "Facenet"
detector_backend = "opencv"     # face detector backend
threshold = 0.4                 # strict cosine similarity threshold

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_dataset_embeddings(dataset_path):
    """Load all face embeddings from dataset folder."""
    known_embeddings = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        embeddings_list = []
        for file in os.listdir(person_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_folder, file)
                try:
                    faces = DeepFace.extract_faces(
                        img_path,
                        detector_backend=detector_backend,
                        enforce_detection=True
                    )
                    for face_obj in faces:
                        face_array = face_obj['face']
                        emb = DeepFace.represent(face_array, model_name=model_name, enforce_detection=False)[0]['embedding']
                        embeddings_list.append(np.array(emb, dtype=np.float32).flatten())
                    print(f"Loaded {file} for {person_name}")
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
        if embeddings_list:
            known_embeddings[person_name] = embeddings_list
    return known_embeddings

def find_closest_face(face_embedding, known_embeddings, threshold=0.9):
    """Find the closest matching person."""
    max_sim = -1
    identity = "Unknown"
    face_embedding = np.array(face_embedding, dtype=np.float32).flatten()
    for name, embeds_list in known_embeddings.items():
        for db_emb in embeds_list:
            db_emb = np.array(db_emb, dtype=np.float32).flatten()
            sim = cosine_similarity(face_embedding, db_emb)
            if sim > max_sim:
                max_sim = sim
                identity = name
    if max_sim < threshold:
        return "Unknown", max_sim
    return identity, max_sim

def get_face_coords(area, face_array):
    """Safely extract bounding box coordinates."""
    if all(k in area for k in ('x','y','w','h')):
        return area['x'], area['y'], area['w'], area['h']
    elif all(k in area for k in ('x1','y1','x2','y2')):
        return area['x1'], area['y1'], area['x2']-area['x1'], area['y2']-area['y1']
    else:
        h, w = face_array.shape[:2]
        return 0, 0, w, h

# ---------------------------
# LOAD DATASET EMBEDDINGS
# ---------------------------
print("Loading known faces from dataset...")
known_embeddings = load_dataset_embeddings(dataset_path)
print(f"Total known people: {len(known_embeddings)}")

# ---------------------------
# START WEBCAM
# ---------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prelim_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(prelim_faces) == 0:
        cv2.imshow("Real-Time Face Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in prelim_faces:
        face_crop = frame[y:y+h, x:x+w]

        try:
            # Extract embedding for the cropped face
            embedding = DeepFace.represent(face_crop, model_name=model_name, enforce_detection=True)[0]['embedding']
            embedding = np.array(embedding, dtype=np.float32).flatten()

            # Compare with known embeddings
            name, sim = find_closest_face(embedding, known_embeddings, threshold)
            print(f"Similarity: {sim:.3f} -> Recognized: {name}")

            # Draw rectangle and label
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print("Face detection failed:", e)

    cv2.imshow("Real-Time Face Recognition", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
