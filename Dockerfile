FROM python:3.12

RUN groupadd -g 1001 localuser && useradd -u 1001 -g localuser -m user

USER user

WORKDIR /app

COPY main.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
