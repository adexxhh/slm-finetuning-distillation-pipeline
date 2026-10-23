FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY . .
RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["python","-m","uvicorn","serving.app:app","--host","0.0.0.0","--port","8000"]
