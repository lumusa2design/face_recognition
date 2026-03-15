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
        if name.lower() == person_name.lower():
            return int(label_id), labels

    if labels:
        new_id = max(int(k) for k in labels.keys()) + 1
    else:
        new_id = 0

    labels[str(new_id)] = person_name
    save_labels(labels)
    return new_id, labels


def get_existing_label(person_name):
    labels = load_labels()
    for label_id, name in labels.items():
        if name.lower() == person_name.lower():
            return int(label_id), labels
    return None, labels


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


def is_blurry(img, threshold=100.0):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold


def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (200, 200))
    face_equalized = cv2.equalizeHist(face_resized)
    return face_equalized


def get_next_image_index(person_dir, label_id):
    if not os.path.exists(person_dir):
        return 0

    max_index = -1
    prefix = f"{label_id}_"

    for file_name in os.listdir(person_dir):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if not file_name.startswith(prefix):
            continue

        name_without_ext = os.path.splitext(file_name)[0]
        parts = name_without_ext.split("_")

        if len(parts) != 2:
            continue

        try:
            idx = int(parts[1])
            max_index = max(max_index, idx)
        except ValueError:
            continue

    return max_index + 1


def create_recognizer():
    if not hasattr(cv2, "face"):
        return None

    return cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )


def capture_photos(person_name, label_id, target=30):
    ensure_dirs()

    person_dir = os.path.join(PEOPLE_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    start_index = get_next_image_index(person_dir, label_id)
    saved_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    print(f"[INFO] Captura iniciada para: {person_name}")
    print("[INFO] Pulsa 's' para guardar una foto cuando la cara esté bien detectada.")
    print("[INFO] Pulsa 'q' para salir.")
    print("[INFO] Consejo: guarda fotos con distintos ángulos, luces y expresiones.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_largest_face(gray, face_cascade)

        display = frame.copy()

        cv2.putText(
            display,
            f"Persona: {person_name}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display,
            f"Guardadas en esta sesion: {saved_count}/{target}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            display,
            "Teclas: s = guardar | q = salir",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        face_processed = None
        blur_text = "N/A"

        if face is not None:
            x, y, w, h = face
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = gray[y:y + h, x:x + w]
            face_processed = preprocess_face(face_roi)

            blurry = is_blurry(face_processed)
            blur_text = "BORROSA" if blurry else "OK"

            cv2.putText(
                display,
                f"Cara detectada | Calidad: {blur_text}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if not blurry else (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                display,
                "No se detecta ninguna cara",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Captura de fotos", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            if face_processed is None:
                print("[WARN] No hay una cara válida para guardar.")
                continue

            if is_blurry(face_processed):
                print("[WARN] Foto borrosa, no se guarda.")
                continue

            img_index = start_index + saved_count
            img_path = os.path.join(person_dir, f"{label_id}_{img_index:03d}.jpg")
            cv2.imwrite(img_path, face_processed)
            saved_count += 1
            print(f"[OK] Guardada: {img_path}")

            if saved_count >= target:
                print("[INFO] Objetivo de fotos alcanzado.")
                break

    cap.release()
    cv2.destroyAllWindows()


def register_new_person():
    ensure_dirs()
    person_name = input("Nombre de la persona nueva: ").strip()

    if not person_name:
        print("[ERROR] Nombre vacío.")
        return

    existing_id, _ = get_existing_label(person_name)
    if existing_id is not None:
        print(f"[WARN] La persona '{person_name}' ya existe.")
        print("[INFO] Usa la opción de añadir fotos a persona existente.")
        return

    label_id, _ = get_or_create_label(person_name)
    capture_photos(person_name, label_id, target=30)


def add_photos_to_existing_person():
    ensure_dirs()
    person_name = input("Nombre de la persona existente: ").strip()

    if not person_name:
        print("[ERROR] Nombre vacío.")
        return

    label_id, _ = get_existing_label(person_name)
    if label_id is None:
        print(f"[ERROR] La persona '{person_name}' no existe.")
        print("[INFO] Primero debes registrarla.")
        return

    try:
        target = int(input("¿Cuántas fotos quieres añadir? ").strip())
        if target <= 0:
            print("[ERROR] El número debe ser mayor que 0.")
            return
    except ValueError:
        print("[ERROR] Debes introducir un número válido.")
        return

    capture_photos(person_name, label_id, target=target)


def list_people():
    labels = load_labels()
    if not labels:
        print("[INFO] No hay personas registradas.")
        return

    print("\n=== PERSONAS REGISTRADAS ===")
    for label_id, name in sorted(labels.items(), key=lambda x: int(x[0])):
        person_dir = os.path.join(PEOPLE_DIR, name)
        img_count = 0

        if os.path.exists(person_dir):
            img_count = len([
                f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        print(f"{label_id}: {name} ({img_count} fotos)")


def train_model():
    ensure_dirs()

    recognizer = create_recognizer()
    if recognizer is None:
        print("[ERROR] Tu OpenCV no tiene el módulo 'cv2.face'.")
        print("[INFO] Instala opencv-contrib-python y no solo opencv-python.")
        return

    labels = load_labels()

    if not labels:
        print("[ERROR] No hay etiquetas registradas.")
        return

    faces = []
    ids = []
    skipped_blurry = 0
    skipped_small = 0
    skipped_invalid = 0

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
                skipped_invalid += 1
                continue

            if img.shape[0] < 100 or img.shape[1] < 100:
                skipped_small += 1
                continue

            img = preprocess_face(img)

            if is_blurry(img):
                skipped_blurry += 1
                continue

            faces.append(img)
            ids.append(int(label_id))

    if not faces:
        print("[ERROR] No hay imágenes válidas para entrenar.")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_FILE)

    print(f"[INFO] Modelo entrenado y guardado en: {MODEL_FILE}")
    print(f"[INFO] Total de imágenes usadas: {len(faces)}")
    print(f"[INFO] Imágenes descartadas por blur: {skipped_blurry}")
    print(f"[INFO] Imágenes descartadas por tamaño: {skipped_small}")
    print(f"[INFO] Imágenes inválidas: {skipped_invalid}")


def recognize():
    recognizer = create_recognizer()
    if recognizer is None:
        print("[ERROR] Tu OpenCV no tiene el módulo 'cv2.face'.")
        print("[INFO] Instala opencv-contrib-python y no solo opencv-python.")
        return

    if not os.path.exists(MODEL_FILE):
        print("[ERROR] No existe el modelo. Entrena primero.")
        return

    labels = load_labels()
    labels = {int(k): v for k, v in labels.items()}

    recognizer.read(MODEL_FILE)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    # Ajusta esto según tus pruebas:
    # más bajo = más estricto
    # más alto = más permisivo
    CONFIDENCE_THRESHOLD = 55

    print("[INFO] Reconocimiento iniciado. Pulsa 'q' para salir.")
    print(f"[INFO] Umbral actual: {CONFIDENCE_THRESHOLD}")

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
            face_roi = gray[y:y + h, x:x + w]
            face_processed = preprocess_face(face_roi)

            if is_blurry(face_processed):
                color = (0, 165, 255)
                text = "Cara borrosa"
            else:
                label_id, confidence = recognizer.predict(face_processed)
                print(f"[DEBUG] label={label_id}, confidence={confidence:.2f}")

                if confidence < CONFIDENCE_THRESHOLD:
                    name = labels.get(label_id, "Desconocido")
                    color = (0, 255, 0)
                    text = f"{name} ({confidence:.1f})"
                else:
                    color = (0, 0, 255)
                    text = f"Desconocido ({confidence:.1f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Reconocimiento facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def delete_person():
    labels = load_labels()
    if not labels:
        print("[INFO] No hay personas registradas.")
        return

    list_people()
    person_name = input("Nombre exacto de la persona a borrar: ").strip()

    if not person_name:
        print("[ERROR] Nombre vacío.")
        return

    label_id, labels = get_existing_label(person_name)
    if label_id is None:
        print("[ERROR] Esa persona no existe.")
        return

    confirm = input(f"¿Seguro que quieres borrar a '{person_name}'? (s/n): ").strip().lower()
    if confirm != "s":
        print("[INFO] Operación cancelada.")
        return

    person_dir = os.path.join(PEOPLE_DIR, person_name)
    if os.path.exists(person_dir):
        for file_name in os.listdir(person_dir):
            file_path = os.path.join(person_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        try:
            os.rmdir(person_dir)
        except OSError:
            pass

    labels.pop(str(label_id), None)
    save_labels(labels)

    print(f"[INFO] Persona '{person_name}' eliminada.")
    if os.path.exists(MODEL_FILE):
        print("[INFO] Recuerda volver a entrenar el modelo.")


def menu():
    while True:
        print("\n=== MENU ===")
        print("1. Registrar persona nueva")
        print("2. Añadir fotos a persona existente")
        print("3. Ver personas registradas")
        print("4. Entrenar modelo")
        print("5. Reconocer")
        print("6. Borrar persona")
        print("7. Salir")

        option = input("Elige una opción: ").strip()

        if option == "1":
            register_new_person()
        elif option == "2":
            add_photos_to_existing_person()
        elif option == "3":
            list_people()
        elif option == "4":
            train_model()
        elif option == "5":
            recognize()
        elif option == "6":
            delete_person()
        elif option == "7":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    ensure_dirs()
    menu()