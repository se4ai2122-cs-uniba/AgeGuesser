FROM node:14

WORKDIR /usr/src/app

COPY . .

RUN npm install

RUN npm run build

EXPOSE 80

CMD [ "npm","run", "start-prod"]