FROM python:3.12

RUN groupadd -g 1001 localuser && useradd -u 1001 -g localuser -m user

USER user

WORKDIR /app

COPY requirements.txt .
COPY resource_monitor.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -i https://test.pypi.org/simple/ runapy==3.0.0

CMD ["python", "BNY-metrics.py"]
