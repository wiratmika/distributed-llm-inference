FROM python:3.11-slim
WORKDIR /app

ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HOME=/opt/hf_cache
RUN mkdir -p /opt/hf_cache && chown -R root:root /opt/hf_cache

RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

ARG MODEL_NAME=gpt2-xl
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('${MODEL_NAME}'); \
AutoTokenizer.from_pretrained('${MODEL_NAME}');"

COPY src/ ./src/

# To be replaced by Docker Compose
CMD ["python", "src/gateway.py"]
