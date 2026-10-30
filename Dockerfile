FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv
WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/srv/.venv/bin:$PATH" \
    DATA_DIR=/srv/data ARTIFACT_DIR=/srv/artifacts
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# Train at build time from the committed CSVs; the model artifact never goes into git.
ARG TRAIN_NROWS=
COPY data/train_features.csv data/train_salaries.csv ./data/
RUN python -m salary.train ${TRAIN_NROWS:+--nrows $TRAIN_NROWS}
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=5s --retries=20 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "salary.api:app", "--host", "0.0.0.0", "--port", "8000"]
