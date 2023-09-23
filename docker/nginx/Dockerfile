# production environment
FROM nginx:1.15-alpine

RUN chown nginx:nginx /usr/share/nginx/html/*

COPY nginx.conf /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]