FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/

COPY data/processed/citation_lens.db /app/data/processed/citation_lens.db
COPY data/processed/cached_titles.json /app/data/processed/cached_titles.json
COPY data/processed/demo_catalog.json /app/data/processed/demo_catalog.json

COPY models/ /app/models/

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]