FROM python:3.12-slim AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --compile-bytecode

FROM dependencies AS training
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
COPY train.py ./
COPY data/online_shoppers_intention.csv ./data/
RUN .venv/bin/python train.py

FROM python:3.12-slim AS service
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PORT=9696 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY --from=dependencies /app/.venv ./.venv
COPY --from=training /app/artifacts/model.joblib ./artifacts/
COPY predict.py ./
COPY examples/session.json ./examples/
USER 10001
EXPOSE 9696
CMD ["python", "predict.py"]
