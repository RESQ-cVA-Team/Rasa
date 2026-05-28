FROM rasa/rasa:3.6.21@sha256:7c0204065d4859e1b7a691c972ca3d26f5d39ad23fbd992b654084721226d813

ENV RASA_TELEMETRY_ENABLED=false
ENV SQLALCHEMY_SILENCE_UBER_WARNING=1
ENV PYTHONPATH=/app:/app/src

ARG LAYERS
ENV LAYERS=${LAYERS}

USER root

WORKDIR /app

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

# Always run with API, native token auth, bounded timeouts, and the core endpoints file.
ENTRYPOINT ["sh", "-lc", "test -n \"${RASA_AUTH_TOKEN:-}\" || { echo 'RASA_AUTH_TOKEN is required' >&2; exit 1; }; exec rasa run --enable-api --auth-token \"$RASA_AUTH_TOKEN\" --endpoints src/core/endpoints.yml --request-timeout 300 --response-timeout 300"]
CMD []
