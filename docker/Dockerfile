# ── base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by FAISS, PyTorch (CPU), and umap-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── source ────────────────────────────────────────────────────────────────────
COPY src/       ./src/
COPY api/       ./api/
COPY app/       ./app/
COPY data/      ./data/

# ── API stage ─────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Streamlit stage ───────────────────────────────────────────────────────────
FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
