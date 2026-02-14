# Distributed LLM Inference Experiment

## Local Installation

1. Use Python 3.11 as it provides universal compatibility; pyenv is recommended.
2. Install Poetry `pipx install poetry`
3. Install dependencies `poetry install`
4. Run `launch_local.sh`

### Sending Request
```
curl -X POST http://localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Hello world!"}'
```

### Configure Model and Worker Nodes
```
./scripts/launch_local.sh              # default: 3 worker nodes gpt2 model
./scripts/launch_local.sh 4            # 4 worker nodes
./scripts/launch_local.sh 2 gpt2       # 2 nodes, gpt2 model
```
