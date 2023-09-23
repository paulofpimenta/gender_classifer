# production environment
FROM nginx:alpine

RUN chmod -R 755 /usr/share/nginx/html/*

COPY nginx.conf /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]