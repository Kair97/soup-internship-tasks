FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY app ./app
COPY chaos ./chaos
COPY tests ./tests
COPY pytest.ini ./
# Default command = API; the worker/chaos services override it.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
