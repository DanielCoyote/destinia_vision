# Dockerfile
FROM python:3.10-slim

# 1. Directorio de trabajo
WORKDIR /app

# 2. Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia el resto del código
COPY . .

# 4. Expone el puerto
EXPOSE 8000

# 5. Comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
