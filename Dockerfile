FROM python:3.12

RUN groupadd -g 1001 localuser && useradd -u 1001 -g localuser -m user

USER user

WORKDIR /app

COPY requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

# Add default environment variable
ENV OUTPUT_DIR=/mnt/data

CMD ["python", "main.py"]
