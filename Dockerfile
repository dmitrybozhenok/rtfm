FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY schemas/ schemas/

RUN pip install --no-cache-dir .

COPY docs/ docs/

EXPOSE 8000

CMD ["uvicorn", "rtfm.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
