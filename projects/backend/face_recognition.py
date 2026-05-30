import cv2
import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

os.makedirs(FACES_DIR, exist_ok=True)

# --------------------------------------------------
# Face Detector
# --------------------------------------------------

cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# --------------------------------------------------
# Detect Faces
# --------------------------------------------------

def extract_faces(img):
    """
    Detect faces from an image.

    Returns:
        list of face coordinates:
        (x, y, w, h)
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces

# --------------------------------------------------
# Train KNN Model
# --------------------------------------------------

def train_model():
    """
    Train face recognition model
    using images inside static/faces
    """

    faces = []
    labels = []

    if not os.path.exists(FACES_DIR):
        raise FileNotFoundError(
            f"Faces directory not found: {FACES_DIR}"
        )

    for user in os.listdir(FACES_DIR):

        user_path = os.path.join(
            FACES_DIR,
            user
        )

        if not os.path.isdir(user_path):
            continue

        print(f"Training user: {user}")

        for image_name in os.listdir(user_path):

            image_path = os.path.join(
                user_path,
                image_name
            )

            img = cv2.imread(image_path)

            if img is None:
                print(
                    f"Skipping invalid image: {image_path}"
                )
                continue

            try:

                resized_face = cv2.resize(
                    img,
                    (50, 50)
                )

                flattened_face = resized_face.flatten()

                faces.append(flattened_face)

                labels.append(user)

            except Exception as e:

                print(
                    f"Error processing {image_path}: {e}"
                )

    if len(faces) == 0:
        raise ValueError(
            "No valid training images found."
        )

    n_neighbors = min(
        5,
        len(faces)
    )

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    knn.fit(
        np.array(faces),
        np.array(labels)
    )

    joblib.dump(
        knn,
        MODEL_PATH
    )

    print(
        f"Model trained successfully."
    )

    print(
        f"Total Faces: {len(faces)}"
    )

    print(
        f"Total Users: {len(set(labels))}"
    )

    return {
        "success": True,
        "total_faces": len(faces),
        "total_users": len(set(labels)),
        "model_path": MODEL_PATH
    }

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

def load_model():
    """
    Load saved face recognition model
    """

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Face recognition model not found. Train the model first."
        )

    return joblib.load(MODEL_PATH)

# --------------------------------------------------
# Predict Face
# --------------------------------------------------

def predict_face(face_image):
    """
    Predict person from face image
    """

    model = load_model()

    resized_face = cv2.resize(
        face_image,
        (50, 50)
    ).flatten()

    prediction = model.predict(
        [resized_face]
    )[0]

    return prediction