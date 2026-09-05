#!/bin/bash
# Stops the Prometheus/Grafana pair started by scripts/observability_start.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/dashboard/.runtime"

for name in prometheus grafana; do
    pidfile="$RUNTIME_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        pid="$(cat "$pidfile")"
        if kill "$pid" 2>/dev/null; then
            echo "stopped $name (pid $pid)"
        else
            echo "$name (pid $pid) was not running"
        fi
        rm -f "$pidfile"
    else
        echo "no pidfile for $name — not started via observability_start.sh?"
    fi
done
