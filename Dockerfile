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

# OS packages get security patches independently of the pinned image tag.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# rasa==3.6.21 pins tensorflow==2.12.0, skops==0.9.0, and protobuf<4.23.4
# exactly -- none of those three (nor keras, tensorflow's own dependency,
# locked to it in lockstep) can move without breaking compatibility rasa
# itself was never tested against. The thirteen below aren't pinned that
# tightly (rasa requires python-engineio!=5.0.0,<6,>=4;
# python-socketio<6,>=4.4; ujson<6.0,>=1.35; PyJWT[crypto]<3.0.0,>=2.0.0;
# aiohttp<3.10,>=3.9.0; cryptography>=41.0.7 -- msgpack/pyasn1/urllib3/
# Pillow/Werkzeug/fonttools/grpcio aren't rasa's own constraints at all,
# just transitive deps) -- bumping them to a fixed release still satisfies
# every declared constraint, confirmed via a dry-run install against this
# exact image before adding this step, and via a full rebuild + real server
# boot afterward.
#
# wheel's own CVE fix (0.46.2) is deliberately NOT bumped here: it's the
# first release requiring packaging>=24.0, and rasa's own
# rasa.shared.utils.validation still imports packaging.version.LegacyVersion,
# removed in packaging 22.0 -- bumping wheel breaks the container outright
# (confirmed by trying it). wheel is a build-time tool anyway; nothing at
# this container's actual runtime processes untrusted .whl files, so the
# residual risk is low. Tracked for suppression via .trivyignore instead.
#
# aiohttp is deliberately left at 3.9.4 despite newer CVEs: every available
# fix (3.13.3+) requires aiohttp>=3.10, which violates rasa's own
# aiohttp<3.10 pin -- same unfixable category as tensorflow/skops/protobuf
# above. Tracked via .trivyignore instead.
RUN pip install --no-cache-dir --upgrade \
	python-engineio==4.13.2 \
	python-socketio==5.16.2 \
	ujson==5.12.1 \
	urllib3==2.7.0 \
	msgpack==1.2.1 \
	pyasn1==0.6.4 \
	Pillow==12.3.0 \
	PyJWT==2.13.0 \
	Werkzeug==3.0.3 \
	aiohttp==3.9.4 \
	cryptography==50.0.1 \
	fonttools==4.43.0 \
	grpcio==1.56.2

WORKDIR /app

RUN mkdir -p /app/.data && chown -R 1001:1001 /app/.data && chmod 700 /app/.data

COPY --chown=1001:1001 src/ src/
COPY --chown=1001:1001 scripts/ scripts/

RUN chmod +x /app/scripts/*.sh

# Ensure local 'src' is a real package to shadow any site-packages 'src'
RUN test -f /app/src/__init__.py || echo "# project package root" > /app/src/__init__.py \
	&& chown 1001:1001 /app/src/__init__.py

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
