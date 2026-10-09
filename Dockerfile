FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY modelo/ ./modelo/
COPY data/fipe_api.csv ./data/fipe_api.csv

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]