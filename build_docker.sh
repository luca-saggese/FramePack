#!/bin/bash
docker build -t framepack .
echo "✅ Build completata!"
echo "👉 Per eseguire il container usa:"
echo "docker run --rm -it --gpus all -p 8080:8080 -v /home/lvx/huggingface:/huggingface framepack"