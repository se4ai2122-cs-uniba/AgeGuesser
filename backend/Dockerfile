FROM python:3.9-slim-buster

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt 

WORKDIR /app
COPY . .

EXPOSE 80

CMD ["gunicorn","-k", "uvicorn.workers.UvicornWorker", "-w", "1", "app.main:app", "-b", "0.0.0.0:80"]