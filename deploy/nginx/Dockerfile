# Pull nginx image from DockerHub; nginx
FROM nginx

# Remove default nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Replace nginx config with a modified version
COPY nginx.conf /etc/nginx/conf.d/
