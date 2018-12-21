#!/bin/bash
set -e

record_experiment_status() {
    if [ "$COMPUTER_TYPE" = 'server' ]; then # Record the experiment status only when running on the server
        STATUS_DIR="${RESULTS_DIR}/status"
        mkdir -p "$STATUS_DIR"

        STATUS_FILE="${STATUS_DIR}/$1"
        if [ ! -f "$STATUS_FILE" ]; then
            # Current local date and time, in ISO 8601 format (microseconds and TZ removed)
            echo `date +%FT%T` > $STATUS_FILE
        fi
    fi
}

if [ "$1" = 'supervisord' ]; then # Normal startup, i.e. no explicit command passed to 'docker run'
    record_experiment_status 'init'

    if [ -n "$DOCKER_CONFIG_SCRIPT" ]; then
        SCRIPT_PATH="/$DOCKER_CONFIG_SCRIPT"
        echo "Running the configuration script $SCRIPT_PATH..."
        bash -c "$SCRIPT_PATH"
    fi

    record_experiment_status 'configured'

    exec "$@"
fi

exec "$@"
