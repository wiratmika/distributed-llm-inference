#!/bin/bash

set -eo pipefail

REPO_URL="https://github.com/wiratmika/distributed-llm-inference.git"
PROJECT_DIR='$HOME/distributed-llm-inference'
ZONE="us-west1-a"

run_on_vm() {
  local vm_name=$1
  local command=$2
  
  gcloud compute ssh "$vm_name" \
    --zone="$ZONE" \
    --command="$command"
}

ALL_VMS="gateway worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7"

for vm in $ALL_VMS; do  
  run_on_vm "$vm" "
    PYTHON_BIN=\$(command -v python3.11 || command -v python3)
    sudo ln -sf \"\$PYTHON_BIN\" /usr/bin/python3 || true
    
    if [ -d \"$PROJECT_DIR\" ]; then
      cd \"$PROJECT_DIR\" && git pull
    else
      git clone $REPO_URL \"$PROJECT_DIR\"
    fi
    
    cd \"$PROJECT_DIR\"
    /usr/local/bin/poetry env use \"\$PYTHON_BIN\"
    /usr/local/bin/poetry install --no-root
    
    pkill -f uvicorn || true
  " &
done

wait

get_ip() {
  gcloud compute instances describe "$1" --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)'
}

echo "Internal IPs:"
G1_IP=$(get_ip gateway)

W1_IP=$(get_ip worker-1)
W2_IP=$(get_ip worker-2)
W3_IP=$(get_ip worker-3)
W4_IP=$(get_ip worker-4)
W5_IP=$(get_ip worker-5)
W6_IP=$(get_ip worker-6)
W7_IP=$(get_ip worker-7)

start_service() {
  local vm=$1
  local env_vars=$2
  local entry=$3
  local port=$4
  
  run_on_vm "$vm" "cd $PROJECT_DIR && nohup env $env_vars /usr/local/bin/poetry run uvicorn $entry --host 0.0.0.0 --port $port > app.log 2>&1 &"
}

# Group 1: 1 worker
start_service "worker-1" "NUM_NODES=1 RANK=0" "inference.worker:app" 8001

# Group 2: 2 workers
start_service "worker-3" "NUM_NODES=2 RANK=1" "inference.worker:app" 8003
start_service "worker-2" "NUM_NODES=2 RANK=0 NEXT_NODE_URL=http://$W3_IP:8003" "inference.worker:app" 8002

# Group 3: 4 workers
start_service "worker-7" "NUM_NODES=4 RANK=3" "inference.worker:app" 8007
start_service "worker-6" "NUM_NODES=4 RANK=2 NEXT_NODE_URL=http://$W7_IP:8007" "inference.worker:app" 8006
start_service "worker-5" "NUM_NODES=4 RANK=1 NEXT_NODE_URL=http://$W6_IP:8006" "inference.worker:app" 8005
start_service "worker-4" "NUM_NODES=4 RANK=0 NEXT_NODE_URL=http://$W5_IP:8005" "inference.worker:app" 8004

# Single global gateway routes requests based on requested node count (1/2/4)
start_service "gateway" "WORKER_URL_1=http://$W1_IP:8001 WORKER_URL_2=http://$W2_IP:8002 WORKER_URL_4=http://$W4_IP:8004" "inference.gateway:app" 8000
