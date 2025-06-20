# Usa una imagen base de Python 3.10
FROM python:3.10-slim

# 1) Directorio de trabajo
WORKDIR /app

# 2) Instala las librerías de sistema necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 3) Copia e instala las dependencias Python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copia todo el código al contenedor
COPY . .

# 5) Expone el puerto para Uvicorn
EXPOSE 8000

# 6) Comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
