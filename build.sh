docker buildx build --platform linux/amd64 -t wiratmika/distributed-llm-inference-gpt2 .
docker buildx build --platform linux/amd64 -t wiratmika/distributed-llm-inference-gpt2-xl --build-arg MODEL_NAME=gpt2-xl .
