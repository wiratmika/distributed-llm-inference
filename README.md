# Distributed LLM Inference Experiment

Note: this research is still a work in progress and aims to be completed by mid-March 2026.

## Background

Many modern large language models (LLMs) contain billions of parameters. Due to its size, running inference using large models such as LLaMA 3.1 405B and Qwen2-72B may not be feasible with a single node, necessitating sharding across multiple nodes. This project aims to create a distributed inference system using open-weight models by partitioning the model layers across distributed nodes. The method proposed is distributed model parallelism. The core idea is to take a transformer-based model and split its layers across multiple machines.

<p align="center">
  <img src="./assets/layer-partitioning.svg" width="500">
</p>
<p align="center">
  <sub><b>Figure 1:</b> Neural network layer partitioning</sub>
</p>

Each node runs its chunk of the model on the input activations and then ships the resulting tensor to the next node. Multiple tokens can be in different pipeline stages at once. This setup will theoretically allow fitting larger models with smaller machines and increase overall throughput.

<p align="center">
  <img src="./assets/parallel-processing.svg" width="500">
</p>
<p align="center">
  <sub><b>Figure 1:</b> Parallel token processing</sub>
</p>

The primary goals of this study are to learn how to create such a system from scratch and benchmark the performance, scalability, and its trade-offs. Due to its distributed nature, the primary penalty is the latency caused by communication overhead between nodes. In addition, challenges arise from scheduling complexity, fault tolerance, straggler (slow) nodes, error propagation, and debugging.

The emphasis of this project are the benefits and trade-offs of distributed computing in model inference, and optimizing for maximum performance is not part of the design goal. Therefore, to simplify the deployment environment and save costs, the model will run using CPU only and will not utilize any GPU. By not requiring a GPU, the setup can be easily replicated with generic cloud virtual machines.

With that constraint in mind, this study is using GPT-2 family models. It has simple, well-understood architecture making it simple enough to understand every component but complex enough to reveal real distributed systems challenges. There are also non-technical practical benefits, such as its mature ecosystem with excellent documentation and permissive license (MIT).

Specifically, the proof-of-concept will use GPT-2 Small to build the infrastructure and validate the architecture works before scaling. It has 124M parameters and 12 layers that are easy to split, test, and runs very fast even on CPU. Eventually, the experiment will use GPT-2 XL, as the size is large enough that distributed inference makes sense in that it will not fit easily in a single CPU memory with full context. At 1.5B parameters and 48 layers, the size will see real benefits from sharding without needing excessive infrastructure.

Inter-process communications is using HTTP as the overhead is extremely small compared to the inference latency.

## Local Installation

1. Use Python 3.11 as it provides universal compatibility; pyenv is recommended.
2. Install Poetry `pipx install poetry`
3. Install dependencies `poetry install --no-root`
4. Run `./launch_local.sh`

### Sending request
```
curl -X POST http://localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Hello world!"}'
```

### Configure model and worker nodes
```
./launch_local.sh              # default: 3 worker nodes, gpt2 model
./launch_local.sh 4            # 4 worker nodes
./launch_local.sh 2 gpt2-xl    # 2 nodes, gpt2-xl model
```

## Planned Research Variables

### Number of nodes
1: Baseline (no distribution)
2: Simple split
3: Balanced split (primary configuration)
4: Finer granularity

### Input sequence length
16: Very short - minimal communication overhead
64: Short conversation
128: Medium (primary benchmark length)
256: Long paragraph
512: Very long
1024: Maximum for GPT-2

### Batch size
1: Single request (baseline - no pipeline benefit)
4: Small batch - some pipeline filling
8: Medium batch (primary benchmark size)
16: Large batch - good pipeline utilization
