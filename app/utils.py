import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from time import time
from mediapipe.python.solutions.face_mesh import FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, FACEMESH_LIPS

# ========== Cargar modelos ==========
def load_tflite_model(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    return interp, input_details, output_details

eye_interp, eye_input, eye_output = load_tflite_model("app/modelos/eye_model_int8.tflite")
mouth_interp, mouth_input, mouth_output = load_tflite_model("app/modelos/mouth_model_int8.tflite")

# ========== Inicializar FaceMesh ==========
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LEFT_EYE_IDXS  = sorted({idx for pair in FACEMESH_LEFT_EYE  for idx in pair})
RIGHT_EYE_IDXS = sorted({idx for pair in FACEMESH_RIGHT_EYE for idx in pair})
MOUTH_IDXS     = sorted({idx for pair in FACEMESH_LIPS      for idx in pair})

# ========== Estado persistente ==========
estado_usuario = {
    "tiempo_ojos_cerrados": 0.0,
    "tiempo_bostezo": 0.0,
    "ojos_cerrados_prev": False,
    "bostezo_prev": False,
    "ultima_actualizacion": time(),
}

# ========== Inferencia por región ==========
def predict_tflite(interp, input_detail, output_detail, roi):
    img = cv2.resize(roi, (input_detail['shape'][2], input_detail['shape'][1]))
    if input_detail['dtype'] == np.uint8:
        inp = np.expand_dims(img, axis=0).astype(np.uint8)
    else:
        inp = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    interp.set_tensor(input_detail['index'], inp)
    interp.invoke()
    out = interp.get_tensor(output_detail['index'])[0][0]
    return float(out)

# ========== Análisis de imagen ==========
def analizar_imagen(image_np):
    global estado_usuario
    h, w, _ = image_np.shape
    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    ahora = time()
    delta = ahora - estado_usuario["ultima_actualizacion"]
    estado_usuario["ultima_actualizacion"] = ahora

    if not res.multi_face_landmarks:
        print("No se detectó ningún rostro.")
        # guarda la imagen para depuración
        #cv2.imwrite("debug_no_face.jpg", image_np)

        return {"error": "No se detectó ningún rostro."}

    lm = res.multi_face_landmarks[0].landmark
    resultados = {}

    def extraer_prob(nombre, idxs, interp, inp, out):
        xs = [int(lm[i].x * w) for i in idxs]
        ys = [int(lm[i].y * h) for i in idxs]
        if xs and ys:
            x1, x2 = max(min(xs)-5, 0), min(max(xs)+5, w)
            y1, y2 = max(min(ys)-5, 0), min(max(ys)+5, h)
            roi = image_np[y1:y2, x1:x2]
            if roi.size > 0:
                return predict_tflite(interp, inp, out, roi)
        return None

    # Ojos
    prob_l = extraer_prob("ojo_izquierdo", LEFT_EYE_IDXS, eye_interp, eye_input, eye_output)
    prob_r = extraer_prob("ojo_derecho", RIGHT_EYE_IDXS, eye_interp, eye_input, eye_output)
    ojos_cerrados = (prob_l is not None and prob_l < 0.5) and (prob_r is not None and prob_r < 0.5)

    # Boca
    prob_b = extraer_prob("boca", MOUTH_IDXS, mouth_interp, mouth_input, mouth_output)
    bostezo = prob_b is not None and prob_b > 0.5

    # Acumuladores de tiempo
    if ojos_cerrados:
        estado_usuario["tiempo_ojos_cerrados"] += delta
    else:
        estado_usuario["tiempo_ojos_cerrados"] = 0.0

    if bostezo:
        estado_usuario["tiempo_bostezo"] += delta
    else:
        estado_usuario["tiempo_bostezo"] = 0.0

    # Clasificación del nivel de somnolencia
    nivel = "nulo"
    if estado_usuario["tiempo_ojos_cerrados"] > 1.5:
        nivel = "alto"
    elif estado_usuario["tiempo_bostezo"] > 1.5:
        nivel = "medio"
    elif ojos_cerrados or bostezo:
        nivel = "bajo"

    resultados = {
        "nivel": nivel,
        "tiempo_ojos_cerrados": round(estado_usuario["tiempo_ojos_cerrados"], 2),
        "tiempo_bostezo": round(estado_usuario["tiempo_bostezo"], 2),
        "probabilidades": {
            "ojo_izquierdo": round(prob_l, 2) if prob_l is not None else None,
            "ojo_derecho": round(prob_r, 2) if prob_r is not None else None,
            "boca": round(prob_b, 2) if prob_b is not None else None
        }
    }

    print(f"el nivel de somnolenca es: {nivel}")

    # Guarda la imagen para depuración
    #cv2.imwrite("debug_result.jpg", image_np)

    return resultados
