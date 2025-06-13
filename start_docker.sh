# sudo docker run -it --gpus 1 -v $(pwd) music-generation-app

docker run -it --gpus 1 \
  -v /home/sambyron/engineering/ML/playground/music-generation:/app      \
  -w /app                              \
  music-generation-app:latest                      \
  bash

