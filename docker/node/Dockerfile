FROM node:18-alpine as build

ENV NODE_ENV=production

RUN mkdir -p /usr/src/front

WORKDIR /usr/src/front

COPY ./frontend/package.json ./frontend/yarn.lock /usr/src/front/

RUN yarn install --production --no-progress

COPY ./frontend/ /usr/src/front/

RUN yarn run build

# NGINX
FROM nginx:1.20-alpine
# Install certbot-nginx and make a /etc/letsencrypt folder
RUN apk add python3 python3-dev py3-pip build-base libressl-dev musl-dev libffi-dev rust cargo
RUN pip3 install pip --upgrade
RUN pip3 install certbot-nginx
RUN mkdir /etc/letsencrypt

#Copy build from node to nginx html folder and nginx.conf file
COPY --from=build /usr/src/front/build /usr/share/nginx/html
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]    
