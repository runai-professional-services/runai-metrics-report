FROM python:3.12

RUN groupadd -g 1001 localuser && useradd -u 1001 -g localuser -m user

USER user

WORKDIR /app

COPY requirements.txt .
COPY BNY-metrics.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -i https://test.pypi.org/simple/ runapy

CMD ["python", "BNY-metrics.py"]
