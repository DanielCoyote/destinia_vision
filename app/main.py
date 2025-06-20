from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from .utils import analizar_imagen

app = FastAPI()

@app.post("/analizar")
async def analizar(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    resultado = analizar_imagen(img_np)
    return resultado

# Ejecuta directo desde python con: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


