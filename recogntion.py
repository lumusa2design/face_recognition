import os
import cv2
import time

DATASET_DIR = "data/people"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def is_blurry(frame, threshold=80.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def register_person(person_name, target_photos=20, capture_interval=1.0):
    person_dir = os.path.join(DATASET_DIR, person_name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    saved_count = 0
    last_capture_time = 0

    instructions = [
        "Mira al frente",
        "Gira un poco a la izquierda",
        "Gira un poco a la derecha",
        "Acerca un poco la cara",
        "Alejate un poco",
        "Cambia la expresion"
    ]

    print("[INFO] Registro iniciado")
    print("[INFO] Pulsa 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )

        step_index = min(saved_count // max(1, target_photos // len(instructions)), len(instructions) - 1)
        instruction_text = instructions[step_index]

        cv2.putText(display, f"Registro de: {person_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Instruccion: {instruction_text}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Fotos guardadas: {saved_count}/{target_photos}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = frame[y:y+h, x:x+w]

            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                if not is_blurry(face_roi):
                    filename = os.path.join(person_dir, f"{person_name}_{saved_count:03d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    last_capture_time = current_time
                    print(f"[OK] Guardada: {filename}")
                else:
                    cv2.putText(display, "Foto descartada: borrosa", (20, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif len(faces) > 1:
            cv2.putText(display, "Hay varias caras. Deja solo una.", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "No se detecta ninguna cara.", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Registro facial", display)

        if saved_count >= target_photos:
            print("[INFO] Registro completado")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    nombre = input("Introduce el nombre de la persona a registrar: ").strip()
    if nombre:
        register_person(nombre)