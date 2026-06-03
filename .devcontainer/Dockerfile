FROM rasa/rasa:3.6.21@sha256:7c0204065d4859e1b7a691c972ca3d26f5d39ad23fbd992b654084721226d813

USER root

WORKDIR /workspace

ENV RASA_TELEMETRY_ENABLED=false

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5005

ENTRYPOINT []
CMD ["sleep", "infinity"]
