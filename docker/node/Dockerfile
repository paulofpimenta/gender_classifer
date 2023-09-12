FROM node:18-alpine as build

ENV NODE_ENV=production

RUN mkdir -p /usr/src/front

WORKDIR /usr/src/front

COPY ./frontend/package.json ./frontend/yarn.lock /usr/src/front/

RUN yarn install --production --no-progress

COPY ./frontend/ /usr/src/front/

RUN yarn run build

RUN ls /usr/src/front


FROM nginx:1.15-alpine
COPY --from=build /usr/src/front/build /usr/share/nginx/html
RUN chmod -R 755 /usr/share/nginx/html
