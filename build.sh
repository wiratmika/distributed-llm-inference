docker buildx build --platform linux/amd64 -t wiratmika/distributed-llm-inference:worker -f Dockerfile.worker .
docker buildx build --platform linux/amd64 -t wiratmika/distributed-llm-inference:gateway -f Dockerfile.gateway .
