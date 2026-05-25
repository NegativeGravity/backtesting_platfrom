FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src/backend

WORKDIR /app

COPY requirements*.txt ./

RUN set -eux; \
    python -m pip install --upgrade pip setuptools wheel; \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi; \
    if [ -f requirements-data.txt ]; then pip install -r requirements-data.txt; fi; \
    pip install \
        fastapi \
        "uvicorn[standard]" \
        pandas \
        numpy \
        pydantic \
        pyyaml \
        python-multipart

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]