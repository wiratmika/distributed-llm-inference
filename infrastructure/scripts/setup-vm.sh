#!/bin/bash
set -eo pipefail

apt-get update
apt-get upgrade -y
apt-get install -y software-properties-common git curl wget

if ! apt-get install -y python3.11 python3.11-venv python3.11-dev; then
	add-apt-repository -y ppa:deadsnakes/ppa
	apt-get update
	apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3.11 -
ln -sf /opt/poetry/bin/poetry /usr/local/bin/poetry
