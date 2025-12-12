FROM ghcr.io/prefix-dev/pixi:noble AS build

ARG ENVIRONMENT="cpu"
ARG INTERFACE="k2eg"

# copy source code, pixi.toml and pixi.lock to the container
WORKDIR /app
COPY . .

# install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -qy ca-certificates git

# manually trigger an update of the certificate store
RUN update-ca-certificates

# run the `install` command (or any other). This will also install the dependencies into `/app/.pixi`
RUN pixi install --environment $ENVIRONMENT
# create the shell-hook bash script to activate the environment and run the inference script
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    pixi shell-hook --environment $ENVIRONMENT -s bash >> /app/entrypoint.sh && \
    echo 'main() {' >> /app/entrypoint.sh && \
    echo '  python -m src.online_model.run --interface "$INTERFACE"' >> /app/entrypoint.sh && \
    echo '}' >> /app/entrypoint.sh && \
    echo 'main || {' >> /app/entrypoint.sh && \
    echo '  status=$?' >> /app/entrypoint.sh && \
    echo '  echo "Main command failed with exit code $status."' >> /app/entrypoint.sh && \
    #echo '  exit $status' >> /app/entrypoint.sh && \
    echo '  exec /bin/bash' >> /app/entrypoint.sh && \
    echo '}' >> /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/src/config
ENV EPICS_CA_AUTO_ADDR_LIST=NO

WORKDIR /app

ENTRYPOINT ["/bin/bash"]
