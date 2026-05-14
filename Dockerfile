FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY turbine_mlops ./turbine_mlops
COPY app ./app
COPY train.py ./train.py

RUN python train.py --n-samples 500 --seed 42

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
