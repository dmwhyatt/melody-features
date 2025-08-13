FROM ghcr.io/astral-sh/uv:python3.13-bookworm

COPY requirements.txt .

RUN uv venv
RUN uv pip install -r requirements.txt

COPY . .

RUN uv pip install -e .
