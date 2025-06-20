# Usa una imagen ligera de Python 3.10
FROM python:3.10-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# 1) Instala las librerías de sistema necesarias para OpenCV/CV2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 2) Copia e instala las dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copia el resto de tu código al contenedor
COPY . .

# 4) Expone el puerto que usa Uvicorn
EXPOSE 8000

# 5) Comando de inicio: ajusta si tu app está en otro módulo
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
