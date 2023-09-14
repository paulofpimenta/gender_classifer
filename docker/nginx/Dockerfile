# production environment
FROM nginx:1.15-alpine

WORKDIR /usr/share/nginx/html

RUN rm /etc/nginx/conf.d/default.conf

# Configuration du pare-feu

#RUN iptables -A INPUT -p tcp --dport 80 -j ACCEPT \
#    && iptables -A INPUT -p tcp --dport 443 -j ACCEPT \
#    && iptables -A INPUT -j DROP

# Création d'un utilisateur non privilégié
#RUN adduser -D -H -u 1000 -s /bin/bash nginx

#EXPOSE 443


COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]