####################################
## Dockerfile for live debugging. ##
####################################
FROM node:18
WORKDIR /app
COPY ./Frontend/package*.json ./
RUN npm install
CMD ["npm", "run", "dev"]
