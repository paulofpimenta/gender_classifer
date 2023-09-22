FROM python:3.9

#Set the current working directory

RUN mkdir -p /usr/src/gender_detect
WORKDIR /usr/src/gender_detect

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


COPY ./backend/requirements.txt /usr/src/gender_detect/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /usr/src/gender_detect/requirements.txt

COPY ./backend/app /usr/src/gender_detect
COPY ./model/ /usr/src/gender_detect/model
COPY ./certs/ /etc/letsencrypt

ENV PYTHONPATH="/usr/src/gender_detect"

EXPOSE 8000

CMD ["uvicorn","main:app","--host", "0.0.0.0", "--port", "8000",\
"--ssl-keyfile","/etc/letsencrypt/live/app1.ouicodedata.com/privkey.pem",\
"--ssl-certfile","/etc/letsencrypt/live/app1.ouicodedata.com/fullchain.pem" ]