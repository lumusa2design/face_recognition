import os
import cv2
import json
import numpy as np
import time
import easyocr
from collections import deque, Counter

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

def init_ocr_reader():
    try:
        return easyocr.Reader(['es', 'en'], gpu=False)
    except Exception as e:
        print(f"[WARN] No se pudo inicializar EasyOCR: {e}")
        return None


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_face_regions_for_redness(frame, face_box):
    """
    Regiones aproximadas: frente y mejillas.
    """
    x, y, w, h = face_box
    h_img, w_img = frame.shape[:2]

    def crop_safe(x1, y1, x2, y2):
        x1 = clamp(x1, 0, w_img - 1)
        y1 = clamp(y1, 0, h_img - 1)
        x2 = clamp(x2, 1, w_img)
        y2 = clamp(y2, 1, h_img)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    forehead = crop_safe(
        x + int(w * 0.25), y + int(h * 0.10),
        x + int(w * 0.75), y + int(h * 0.30)
    )

    left_cheek = crop_safe(
        x + int(w * 0.12), y + int(h * 0.45),
        x + int(w * 0.32), y + int(h * 0.68)
    )

    right_cheek = crop_safe(
        x + int(w * 0.68), y + int(h * 0.45),
        x + int(w * 0.88), y + int(h * 0.68)
    )

    return forehead, left_cheek, right_cheek


def compute_redness_score(region):
    """
    Índice simple de enrojecimiento visual.
    """
    if region is None or region.size == 0:
        return None

    mean_b = np.mean(region[:, :, 0])
    mean_g = np.mean(region[:, :, 1])
    mean_r = np.mean(region[:, :, 2])

    score = mean_r - ((mean_g + mean_b) / 2.0)
    return float(score)


def analyze_face_redness(frame, face_box):
    forehead, left_cheek, right_cheek = get_face_regions_for_redness(frame, face_box)

    scores = []
    for region in [forehead, left_cheek, right_cheek]:
        score = compute_redness_score(region)
        if score is not None:
            scores.append(score)

    if not scores:
        return None, "desconocido"

    avg_score = float(np.mean(scores))

    if avg_score < 10:
        level = "bajo"
    elif avg_score < 22:
        level = "medio"
    else:
        level = "alto"

    return avg_score, level


def estimate_apparent_temperature(redness_score):
    """
    Estimación visual MUY aproximada.
    NO es temperatura real ni medición médica.
    """
    if redness_score is None:
        return None, "sin datos"

    # Base arbitraria para “temperatura aparente visual”
    # Ajustable según tus pruebas.
    apparent_temp = 36.0 + (redness_score / 20.0)

    # Limitamos para no decir barbaridades absurdas
    apparent_temp = max(35.0, min(39.5, apparent_temp))

    if apparent_temp < 36.4:
        label = "normal-baja"
    elif apparent_temp < 37.3:
        label = "normal"
    elif apparent_temp < 38.0:
        label = "algo elevada"
    else:
        label = "alta visualmente"

    return apparent_temp, label


def read_text_from_frame(reader, frame):
    if reader is None:
        return []

    try:
        results = reader.readtext(frame)
        texts = []

        for item in results:
            text = item[1].strip()
            conf = item[2]
            if len(text) >= 2 and conf >= 0.35:
                texts.append(text)

        dedup = []
        seen = set()
        for t in texts:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                dedup.append(t)

        return dedup[:5]
    except Exception as e:
        print(f"[WARN] Error en OCR: {e}")
        return []


def describe_environment(face_count, name, ocr_texts, redness_level, apparent_temp, temp_label):
    if face_count == 0:
        people_desc = "No veo ninguna persona"
    elif face_count == 1:
        if name and name != "Desconocido":
            people_desc = f"Veo a {name}"
        else:
            people_desc = "Veo una persona"
    else:
        people_desc = f"Veo {face_count} personas"

    if ocr_texts:
        text_desc = "Texto visible: " + ", ".join(ocr_texts[:3])
    else:
        text_desc = "No veo texto relevante"

    if apparent_temp is not None:
        temp_desc = f"temperatura aparente {apparent_temp:.1f}°C ({temp_label})"
    else:
        temp_desc = "sin estimación aparente"

    return f"{people_desc}. Enrojecimiento {redness_level}. {temp_desc}. {text_desc}."


def fit_frame_to_screen(frame, screen_w=1600, screen_h=900):
    """
    Escala el frame para que aproveche mejor la pantalla sin deformar.
    """
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_multiline_text(img, lines, x=20, y=40, line_height=35,
                        font_scale=0.9, color=(255, 255, 255), thickness=2):
    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    
def get_dominant_bgr_color(region):
    """
    Devuelve el color medio BGR de una región.
    """
    if region is None or region.size == 0:
        return None

    mean_b = float(np.mean(region[:, :, 0]))
    mean_g = float(np.mean(region[:, :, 1]))
    mean_r = float(np.mean(region[:, :, 2]))
    return mean_b, mean_g, mean_r


def classify_color_name_from_bgr(mean_b, mean_g, mean_r):
    """
    Clasificación muy simple de color dominante.
    """
    brightness = (mean_r + mean_g + mean_b) / 3.0

    if brightness < 45:
        return "negra"
    if brightness > 210:
        return "blanca"

    max_channel = max(mean_r, mean_g, mean_b)
    min_channel = min(mean_r, mean_g, mean_b)

    # gris
    if max_channel - min_channel < 18:
        if brightness < 100:
            return "gris oscura"
        elif brightness < 170:
            return "gris"
        else:
            return "gris clara"

    # marrón / beige aproximados
    if mean_r > mean_g > mean_b:
        if brightness < 120:
            return "marrón"
        else:
            return "beige"

    # rojo
    if mean_r > mean_g + 20 and mean_r > mean_b + 20:
        return "roja"

    # verde
    if mean_g > mean_r + 15 and mean_g > mean_b + 15:
        return "verde"

    # azul
    if mean_b > mean_r + 15 and mean_b > mean_g + 15:
        return "azul"

    # amarillo
    if mean_r > 140 and mean_g > 140 and mean_b < 120:
        return "amarilla"

    # rosa / morado aproximados
    if mean_r > 140 and mean_b > 120 and mean_g < 130:
        if mean_r > mean_b:
            return "rosa"
        return "morada"

    # naranja
    if mean_r > 170 and mean_g > 100 and mean_g < mean_r and mean_b < 100:
        return "naranja"

    return "de color indefinido"


def analyze_clothing(frame, face_box):
    """
    Mira una zona debajo de la cara para estimar el color de la ropa superior.
    """
    x, y, w, h = face_box
    h_img, w_img = frame.shape[:2]

    # Región aproximada del torso superior
    tx1 = x - int(w * 0.15)
    tx2 = x + w + int(w * 0.15)
    ty1 = y + h
    ty2 = y + h + int(h * 1.4)

    tx1 = clamp(tx1, 0, w_img - 1)
    tx2 = clamp(tx2, 1, w_img)
    ty1 = clamp(ty1, 0, h_img - 1)
    ty2 = clamp(ty2, 1, h_img)

    if tx2 <= tx1 or ty2 <= ty1:
        return "ropa no visible", None

    torso_region = frame[ty1:ty2, tx1:tx2]
    if torso_region.size == 0:
        return "ropa no visible", None

    mean_color = get_dominant_bgr_color(torso_region)
    if mean_color is None:
        return "ropa no visible", None

    color_name = classify_color_name_from_bgr(*mean_color)
    clothing_desc = f"parte superior {color_name}"

    return clothing_desc, (tx1, ty1, tx2, ty2)


def build_scene_summary_extended(name, face_count, redness_level, apparent_temp, temp_label, ocr_texts, clothing_desc):
    if face_count == 0:
        people_desc = "No veo ninguna persona"
    elif face_count == 1:
        if name and name != "Desconocido":
            people_desc = f"Veo a {name}"
        else:
            people_desc = "Veo una persona"
    else:
        people_desc = f"Veo {face_count} personas"

    if clothing_desc:
        clothing_part = f"lleva {clothing_desc}"
    else:
        clothing_part = "no distingo bien la ropa"

    if apparent_temp is not None:
        temp_part = f"temperatura aparente {apparent_temp:.1f}°C ({temp_label})"
    else:
        temp_part = "sin estimación de temperatura aparente"

    if ocr_texts:
        text_part = "texto visible: " + ", ".join(ocr_texts[:3])
    else:
        text_part = "sin texto relevante visible"

    return f"{people_desc}, {clothing_part}, enrojecimiento {redness_level}, {temp_part}, {text_part}."


def most_common_name(history):
    if not history:
        return "Desconocido"
    counter = Counter(history)
    return counter.most_common(1)[0][0]


def init_ocr_reader():
    """
    Inicializa EasyOCR una sola vez.
    """
    try:
        reader = easyocr.Reader(['es', 'en'], gpu=False)
        return reader
    except Exception as e:
        print(f"[WARN] No se pudo inicializar EasyOCR: {e}")
        return None


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_face_regions_for_redness(frame, face_box):
    """
    Devuelve regiones aproximadas de frente y mejillas en BGR.
    face_box = (x, y, w, h)
    """
    x, y, w, h = face_box

    # Frente
    fx1 = x + int(w * 0.25)
    fy1 = y + int(h * 0.10)
    fx2 = x + int(w * 0.75)
    fy2 = y + int(h * 0.30)

    # Mejilla izquierda
    lx1 = x + int(w * 0.12)
    ly1 = y + int(h * 0.45)
    lx2 = x + int(w * 0.32)
    ly2 = y + int(h * 0.68)

    # Mejilla derecha
    rx1 = x + int(w * 0.68)
    ry1 = y + int(h * 0.45)
    rx2 = x + int(w * 0.88)
    ry2 = y + int(h * 0.68)

    h_img, w_img = frame.shape[:2]

    def crop_safe(x1, y1, x2, y2):
        x1 = clamp(x1, 0, w_img - 1)
        y1 = clamp(y1, 0, h_img - 1)
        x2 = clamp(x2, 1, w_img)
        y2 = clamp(y2, 1, h_img)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    forehead = crop_safe(fx1, fy1, fx2, fy2)
    left_cheek = crop_safe(lx1, ly1, lx2, ly2)
    right_cheek = crop_safe(rx1, ry1, rx2, ry2)

    return forehead, left_cheek, right_cheek


def compute_redness_score(region):
    """
    Calcula una métrica simple de 'enrojecimiento visual'.
    No es temperatura. Solo color relativo.
    """
    if region is None or region.size == 0:
        return None

    mean_b = np.mean(region[:, :, 0])
    mean_g = np.mean(region[:, :, 1])
    mean_r = np.mean(region[:, :, 2])

    # Índice simple: cuánto domina el rojo sobre verde/azul
    score = mean_r - ((mean_g + mean_b) / 2.0)
    return float(score)


def analyze_face_redness(frame, face_box):
    """
    Combina frente y mejillas para sacar un redness score medio.
    """
    forehead, left_cheek, right_cheek = get_face_regions_for_redness(frame, face_box)

    scores = []
    for region in [forehead, left_cheek, right_cheek]:
        score = compute_redness_score(region)
        if score is not None:
            scores.append(score)

    if not scores:
        return None, "desconocido"

    avg_score = float(np.mean(scores))

    # Estos umbrales son orientativos y hay que afinarlos con pruebas
    if avg_score < 10:
        level = "bajo"
    elif avg_score < 22:
        level = "medio"
    else:
        level = "alto"

    return avg_score, level


def read_text_from_frame(reader, frame):
    """
    OCR sobre el frame completo.
    Devuelve una lista de textos cortos detectados.
    """
    if reader is None:
        return []

    try:
        results = reader.readtext(frame)
        texts = []

        for item in results:
            text = item[1].strip()
            conf = item[2]

            if len(text) >= 2 and conf >= 0.35:
                texts.append(text)

        # quitar duplicados conservando orden
        dedup = []
        seen = set()
        for t in texts:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                dedup.append(t)

        return dedup[:5]

    except Exception as e:
        print(f"[WARN] Error en OCR: {e}")
        return []


def build_scene_summary(name, confidence, redness_level, redness_score, ocr_texts):
    """
    Genera una frase simple.
    """
    if name is None:
        person_part = "Veo una persona"
    elif name == "Desconocido":
        person_part = "Veo una persona desconocida"
    else:
        person_part = f"Veo a {name}"

    redness_part = f"enrojecimiento visual {redness_level}"
    if redness_score is not None:
        redness_part += f" ({redness_score:.1f})"

    if ocr_texts:
        text_part = "texto visible: " + ", ".join(ocr_texts[:3])
    else:
        text_part = "sin texto relevante visible"

    return f"{person_part}, {redness_part}, {text_part}."


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

    CONFIDENCE_THRESHOLD = 60

    print("[INFO] Inicializando OCR...")
    ocr_reader = init_ocr_reader()

    window_name = "Reconocimiento facial + entorno"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    print("[INFO] Reconocimiento visual iniciado. Pulsa 'q' para salir.")
    print(f"[INFO] Umbral actual: {CONFIDENCE_THRESHOLD}")

    last_ocr_texts = []
    last_ocr_time = 0
    OCR_INTERVAL = 4.0

    confidence_history = deque(maxlen=10)
    name_history = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(90, 90)
        )

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        face_count = len(faces)

        # Nos quedamos con la cara principal para estabilidad
        faces = faces[:1]

        main_name = None
        main_redness_score = None
        main_redness_level = "desconocido"
        main_apparent_temp = None
        main_temp_label = "sin datos"
        main_clothing_desc = "ropa no visible"

        now = time.time()
        if now - last_ocr_time >= OCR_INTERVAL:
            last_ocr_texts = read_text_from_frame(ocr_reader, frame)
            last_ocr_time = now

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = gray[y:y + h, x:x + w]
            face_processed = preprocess_face(face_roi)

            redness_score, redness_level = analyze_face_redness(frame, (x, y, w, h))
            apparent_temp, temp_label = estimate_apparent_temperature(redness_score)
            clothing_desc, clothing_box = analyze_clothing(frame, (x, y, w, h))

            if is_blurry(face_processed):
                color = (0, 165, 255)
                name = "Cara borrosa"
                text = f"{name} | redness: {redness_level}"
            else:
                label_id, confidence = recognizer.predict(face_processed)
                predicted_name = labels.get(label_id, "Desconocido")

                confidence_history.append(confidence)
                name_history.append(
                    predicted_name if confidence < CONFIDENCE_THRESHOLD else "Desconocido"
                )

                avg_confidence = sum(confidence_history) / len(confidence_history)
                stable_name = most_common_name(name_history)

                if stable_name != "Desconocido" and avg_confidence < CONFIDENCE_THRESHOLD:
                    name = stable_name
                    color = (0, 255, 0)
                else:
                    name = "Desconocido"
                    color = (0, 0, 255)

                temp_text = f"{apparent_temp:.1f}C" if apparent_temp is not None else "--"
                text = (
                    f"{name} | conf media: {avg_confidence:.1f} | "
                    f"redness: {redness_level} | temp aparente: {temp_text}"
                )

            if i == 0:
                main_name = name if name != "Cara borrosa" else None
                main_redness_score = redness_score
                main_redness_level = redness_level
                main_apparent_temp = apparent_temp
                main_temp_label = temp_label
                main_clothing_desc = clothing_desc

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Caja de ropa
            if clothing_box is not None:
                tx1, ty1, tx2, ty2 = clothing_box
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 1)

            cv2.putText(
                frame,
                text,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Ropa: {clothing_desc}",
                (x, min(frame.shape[0] - 20, y + h + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        summary = build_scene_summary_extended(
            main_name,
            face_count,
            main_redness_level,
            main_apparent_temp,
            main_temp_label,
            last_ocr_texts,
            main_clothing_desc
        )

        overlay_lines = [
            f"Personas detectadas: {face_count}",
            f"Persona principal: {main_name if main_name else '--'}",
            f"Ropa detectada: {main_clothing_desc}",
            f"Enrojecimiento: {main_redness_level}",
            f"Temperatura aparente: "
            f"{f'{main_apparent_temp:.1f} C ({main_temp_label})' if main_apparent_temp is not None else '--'}",
            f"OCR: {', '.join(last_ocr_texts[:3]) if last_ocr_texts else '--'}",
            "Resumen:",
            summary[:140],
            "AVISO: la temperatura aparente es solo una estimacion visual, no una medicion real.",
            "Pulsa Q para salir"
        ]

        display = fit_frame_to_screen(frame, screen_w=1920, screen_h=1080)

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 330), (0, 0, 0), -1)
        alpha = 0.48
        display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)

        draw_multiline_text(
            display,
            overlay_lines,
            x=20,
            y=40,
            line_height=32,
            font_scale=0.85,
            color=(255, 255, 255),
            thickness=2
        )

        cv2.imshow(window_name, display)

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