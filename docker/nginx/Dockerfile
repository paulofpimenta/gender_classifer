# production environment
FROM nginx:1.15-alpine

RUN chmod -R 755 /usr/share/nginx/html/*

RUN chown -R nginx:nginx /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]