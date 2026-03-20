# Distributed LLM Inference Experiment

This projects aims to implement CPU-only distributed inference for GPT-2 models using pipeline model parallelism over HTTP.

## Background

Many modern large language models (LLMs) contain billions of parameters. Due to its size, running inference using large models such as LLaMA 3.1 405B and Qwen2-72B may not be feasible with a single node, necessitating sharding across multiple nodes. This project aims to create a distributed inference system using open-weight models by partitioning the model layers across distributed nodes. The method proposed is distributed model parallelism. The core idea is to take a transformer-based model and split its layers across multiple machines.

<p align="center">
  <img src="./assets/layer-partitioning.svg" width="500">
</p>
<p align="center">
  <sub><b>Figure 1:</b> Neural network layer partitioning.</sub>
</p>

Each node runs its chunk of the model on the input activations from a microbatch and then passes the resulting activation tensor to the next node. Multiple microbatches can be processed concurrently in different pipeline stages, allowing the model to be efficiently split across smaller machines and increasing overall throughput.

<p align="center">
  <img src="./assets/parallel-processing.svg" width="500">
</p>
<p align="center">
  <sub><b>Figure 2:</b> Pipeline parallelism. Each microbatch is represented internally as a tensor of activations flowing between pipeline stages.</sub>
</p>

The primary goals of this study are to learn how to create such a system from scratch and benchmark the performance, scalability, and its trade-offs. Due to its distributed nature, the primary penalty is the latency caused by communication overhead between nodes. In addition, challenges arise from scheduling complexity, fault tolerance, straggler (slow) nodes, error propagation, and debugging.

The emphasis of this project are the benefits and trade-offs of distributed computing in model inference, and optimizing for maximum performance is not part of the design goal. Therefore, to simplify the deployment environment and save costs, the model will run using CPU only and will not utilize any GPU. By not requiring a GPU, the setup can be easily replicated with generic cloud virtual machines.

<p align="center">
  <img src="./assets/system-architecture.svg" width="500">
</p>
<p align="center">
  <sub><b>Figure 3:</b> System architecture overview.</sub>
</p>

With that constraint in mind, this study is using GPT-2 family models. It has simple, well-understood architecture making it simple enough to understand every component but complex enough to reveal real distributed systems challenges. There are also non-technical practical benefits, such as its mature ecosystem with excellent documentation and permissive license (MIT).

Specifically, the proof-of-concept will use GPT-2 Small to build the infrastructure and validate the architecture works before scaling. It has 124M parameters and 12 layers that are easy to split, test, and runs very fast even on CPU. Eventually, the experiment will use GPT-2 XL, as the size is large enough that distributed inference makes sense in that it will not fit easily in a single CPU memory with full context. At 1.5B parameters and 48 layers, the size will see real benefits from sharding without needing excessive infrastructure.

Inter-process communications is using HTTP as the overhead is extremely small compared to the inference latency.

## Local Installation

1. Use Python 3.11 as it provides universal compatibility; pyenv is recommended
2. Install Poetry `pipx install poetry`
3. Install dependencies `poetry install --no-root`
4. Run `./launch_local.sh` to launch gateway and worker nodes
5. Run `benchmark-dashboard` to optionally launch benchmarking dashboard

### Sending request
```sh
curl -X POST http://localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Hello world!"}'
```

### Configure model and worker nodes
```sh
./launch_local.sh              # default: 3 worker nodes, gpt2 model
./launch_local.sh 4            # 4 worker nodes
./launch_local.sh 2 gpt2-xl    # 2 nodes, gpt2-xl model
```

## Dashboard
To run benchmarks and aid result visualization, we are using Streamlit dashboard.

```sh
streamlit run benchmark/dashboard.py
```

## Infrastructure Provisioning

1. Install Terraform CLI and Google Cloud CLI, then authenticate
2. Create `infrastructure/terraform.tfvars` with `project_id`, `region`, and `zone`
3. Run 
```
cd infrastructure
terraform init
terraform apply
```
