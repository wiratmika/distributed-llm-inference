#!/bin/bash
set -euo pipefail

MODEL_NAME="${model_name}"
WORKER_URL_1="${worker_url_1}"
WORKER_URL_2="${worker_url_2}"
WORKER_URL_4="${worker_url_4}"
APP_PORT="${app_port}"
REPO_URL="https://github.com/wiratmika/distributed-llm-inference.git"
APP_USER="ubuntu"
PROJECT_DIR="/home/ubuntu/distributed-llm-inference"
SERVICE_NAME="distributed-llm-gateway"

apt-get update
apt-get upgrade -y
apt-get install -y software-properties-common git curl wget

mkdir -p /tmp /var/tmp /usr/tmp
chmod 1777 /tmp /var/tmp /usr/tmp

if ! apt-get install -y python3.11 python3.11-venv python3.11-dev; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3.11 -
ln -sf /opt/poetry/bin/poetry /usr/local/bin/poetry

if [ ! -d "$PROJECT_DIR" ]; then
  sudo -u "$APP_USER" git clone "$REPO_URL" "$PROJECT_DIR"
else
  sudo -u "$APP_USER" bash -lc "cd '$PROJECT_DIR' && git pull"
fi

sudo -u "$APP_USER" bash -lc "cd '$PROJECT_DIR' && /usr/local/bin/poetry env use /usr/bin/python3.11 && /usr/local/bin/poetry install --no-root"
sudo -u "$APP_USER" bash -lc "rm -rf ~/.cache/pypoetry/cache ~/.cache/pypoetry/artifacts ~/.cache/pip || true"

cat >/etc/systemd/system/$SERVICE_NAME.service <<EOF
[Unit]
Description=Distributed LLM Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$PROJECT_DIR
Environment=TMPDIR=/tmp
Environment=MODEL_NAME=$MODEL_NAME
Environment=WORKER_URL_1=$WORKER_URL_1
Environment=WORKER_URL_2=$WORKER_URL_2
Environment=WORKER_URL_4=$WORKER_URL_4
ExecStart=/usr/local/bin/poetry run uvicorn inference.gateway:app --host 0.0.0.0 --port $APP_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME
