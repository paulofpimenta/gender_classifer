FROM python:3.9

#Set the current working directory

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


COPY ./backend/requirements.txt /usr/src/app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /usr/src/app/requirements.txt

COPY ./backend/app /usr/src/app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

