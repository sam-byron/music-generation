# DOCKER_BUILDKIT=0 docker compose -f docker-compose.gpu.yml build --no-cache

# docker build  \
#   -f ./docker Dockerfile.gpu \
#   -t music-gen-gpu .
docker compose -f docker-compose.gpu.yml build --no-cache