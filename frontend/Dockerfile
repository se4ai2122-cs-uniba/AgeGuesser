FROM node:14

WORKDIR /usr/src/app

COPY package.json package.json

RUN npm install

COPY . .

RUN npm run build && rm -rf src scripts rollup.config.js

EXPOSE 80

CMD [ "npm","run", "start-prod"]