#!/bin/bash
set -eo pipefail

apt-get update
apt-get upgrade -y
apt-get install -y software-properties-common git curl wget
apt-get install -y python3.11 python3.11-venv python3.11-dev
curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3.11 -
ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry
