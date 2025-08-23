# FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# COPY requirements.txt .

# RUN uv venv
# RUN uv pip install -r requirements.txt

# COPY . .

# RUN uv pip install -e .

FROM python:3.12.11-bookworm

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter

COPY . .
RUN pip install -e .
RUN chmod +x src/melody_features/install_idyom.sh
RUN ./src/melody_features/install_idyom.sh
