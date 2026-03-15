import os
import cv2
import json
import numpy as np

DATA_DIR = "data"
PEOPLE_DIR = os.path.join(DATA_DIR, "people")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")
MODEL_FILE = os.path.join(DATA_DIR, "trainer.yml")

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def ensure_dirs():
    os.makedirs(PEOPLE_DIR, exist_ok=True)


def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def get_or_create_label(person_name):
    labels = load_labels()

    for label_id, name in labels.items():
        if name == person_name:
            return int(label_id), labels

    if labels:
        new_id = max(int(k) for k in labels.keys()) + 1
    else:
        new_id = 0

    labels[str(new_id)] = person_name
    save_labels(labels)
    return new_id, labels


def detect_largest_face(gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def register_person():
    ensure_dirs()
    person_name = input("Nombre de la persona: ").strip()

    if not person_name:
        print("[ERROR] Nombre vacío.")
        return

    label_id, _ = get_or_create_label(person_name)
    person_dir = os.path.join(PEOPLE_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    count = 0
    target = 30

    print("[INFO] Registro iniciado.")
    print("[INFO] Pulsa 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_largest_face(gray, face_cascade)

        display = frame.copy()

        cv2.putText(display, f"Persona: {person_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Fotos: {count}/{target}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if face is not None:
            x, y, w, h = face
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))

            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_path = os.path.join(person_dir, f"{label_id}_{count:03d}.jpg")
                cv2.imwrite(img_path, face_resized)
                count += 1
                print(f"[OK] Guardada: {img_path}")

        cv2.imshow("Registro", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if count >= target:
            print("[INFO] Registro completado.")
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    ensure_dirs()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = load_labels()

    if not labels:
        print("[ERROR] No hay etiquetas registradas.")
        return

    faces = []
    ids = []

    for label_id, person_name in labels.items():
        person_dir = os.path.join(PEOPLE_DIR, person_name)
        if not os.path.exists(person_dir):
            continue

        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, file_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            ids.append(int(label_id))

    if not faces:
        print("[ERROR] No hay imágenes para entrenar.")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_FILE)
    print(f"[INFO] Modelo entrenado y guardado en {MODEL_FILE}")


def recognize():
    if not os.path.exists(MODEL_FILE):
        print("[ERROR] No existe el modelo. Entrena primero.")
        return

    labels = load_labels()
    labels = {int(k): v for k, v in labels.items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    print("[INFO] Reconocimiento iniciado. Pulsa 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))

            label_id, confidence = recognizer.predict(face_resized)

            # En LBPH, cuanto más bajo, mejor
            if confidence < 70:
                name = labels.get(label_id, "Desconocido")
                color = (0, 255, 0)
                text = f"{name} ({confidence:.1f})"
            else:
                name = "Desconocido"
                color = (0, 0, 255)
                text = f"{name} ({confidence:.1f})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Reconocimiento facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def menu():
    while True:
        print("\n=== MENU ===")
        print("1. Registrar persona")
        print("2. Entrenar modelo")
        print("3. Reconocer")
        print("4. Salir")

        option = input("Elige una opción: ").strip()

        if option == "1":
            register_person()
        elif option == "2":
            train_model()
        elif option == "3":
            recognize()
        elif option == "4":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    ensure_dirs()
    menu()