FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
COPY docker/check-esbuild.mjs docker/
# --ignore-scripts: sin scripts de instalación de terceros. Los binarios nativos (esbuild,
# sharp, Biome) llegan como paquetes opcionales y no los necesitan.
# D03-06: npm omite en silencio un opcional que no pudo descargar; sin el binario de
# esbuild que usa tsx (drizzle-kit migrate, db:seed) el backend muere al arrancar.
# check-esbuild lo detecta: se reintenta una vez y, si sigue faltando, el build falla.
RUN npm ci --ignore-scripts && node docker/check-esbuild.mjs \
  || (echo "[build] reintentando npm ci" && rm -rf node_modules \
      && npm ci --ignore-scripts && node docker/check-esbuild.mjs)

COPY . .
RUN npm run build

RUN chmod +x docker/entrypoint.sh

ENV NODE_ENV=production
EXPOSE 3100

ENTRYPOINT ["docker/entrypoint.sh"]
