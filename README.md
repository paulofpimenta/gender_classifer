1. Introduction

The gender detection application is a full-stack web application that allows live gender detection using the web cam. The model used for gender detection was implmented with PyTorch framework. A CNN was trained using a public human faces dataset of more than 50000 images. 


2. Architecture

* Docker
* Model
* Backend
* Frontend

2.1. Docker

The applicaion is fully dockeried into a container. The docker compose file defines tree images: Web (Node.js and React), API (FastApi with nginx as a reverse proxy), and Certbot for certifications.

Certicates expires in every three months and will be renewed if the docker compose runs again. There are, however, some ways automatize certboot renew via crontab scripts.

2.2. Model

2.3. Backend

2.3. Frontend

3. Installation

4. Limitations
