FROM rasa/rasa:3.6.21@sha256:7c0204065d4859e1b7a691c972ca3d26f5d39ad23fbd992b654084721226d813

ENV RASA_TELEMETRY_ENABLED=false
ENV SQLALCHEMY_SILENCE_UBER_WARNING=1
ENV PYTHONPATH=/app:/app/src

ARG RASA_VERSION=""
ARG RASA_COMMIT_SHA=""
ARG RASA_IMAGE_TAG=""
ARG RASA_BUILD_DATE=""
ARG RASA_SSOT_VERSION=""

ARG LAYERS
ENV RASA_VERSION=${RASA_VERSION}
ENV RASA_COMMIT_SHA=${RASA_COMMIT_SHA}
ENV RASA_IMAGE_TAG=${RASA_IMAGE_TAG}
ENV RASA_BUILD_DATE=${RASA_BUILD_DATE}
ENV RASA_SSOT_VERSION=${RASA_SSOT_VERSION}
ENV LAYERS=${LAYERS}

LABEL org.opencontainers.image.version=${RASA_VERSION}
LABEL org.opencontainers.image.revision=${RASA_COMMIT_SHA}
LABEL org.opencontainers.image.created=${RASA_BUILD_DATE}

USER root

WORKDIR /app

RUN mkdir -p /app/.data && chown -R 1001:1001 /app/.data

COPY src/ src/
COPY scripts/ scripts/

RUN chmod +x /app/scripts/*.sh

# Ensure local 'src' is a real package to shadow any site-packages 'src'
RUN test -f /app/src/__init__.py || echo "# project package root" > /app/src/__init__.py

RUN echo "Using PYTHONPATH=$PYTHONPATH" && \
	python -c "import sys; print('Container sys.path:', sys.path)" && \
	echo 'Listing /app/src:' && ls -la /app/src || true && \
	echo 'Listing /app/src/components:' && ls -la /app/src/components || true && \
	python - <<'PY'
import importlib, sys
print('Precheck sys.path=', sys.path)
try:
	m = importlib.import_module('src')
	print('Imported src from:', getattr(m, '__file__', None))
	print('src.__path__:', getattr(m, '__path__', None))
	cm = importlib.import_module('src.components.layered_importer')
	print('Imported layered_importer from:', getattr(cm, '__file__', None))
except Exception as e:
	print('Import diagnostic error:', repr(e))
	raise
PY

# Run layering + training in a separate step to avoid heredoc chaining issues
RUN PYTHONPATH=/app:/app/src ./scripts/layer_rasa_projects.sh ${LAYERS}

EXPOSE 5005

USER 1001

# Always run through the wrapper; it resolves endpoints from env presets.
ENTRYPOINT ["python3", "-m", "src.run_rasa"]
CMD []
