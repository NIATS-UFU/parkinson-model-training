ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Copy and install pre-built wheel (run `make build` first)
COPY dist/*.whl /tmp/wheels/
RUN uv pip install --system /tmp/wheels/*.whl && rm -rf /tmp/wheels

ENTRYPOINT ["main-cli"]
