#!/bin/bash
# Phase 11: starts a ServeLLM-dedicated Prometheus + Grafana pair as plain
# background processes — this cluster's login node has no Docker/Podman/
# Singularity, so the docker-compose path (docker/docker-compose.yml) isn't
# usable here. Both ship as self-contained static binaries needing neither
# root nor a container runtime.
#
# One-time setup (not run by this script — do this once):
#   mkdir -p ~/tools && cd ~/tools
#   curl -sL -o prometheus.tar.gz "https://github.com/prometheus/prometheus/releases/download/v3.14.0/prometheus-3.14.0.linux-amd64.tar.gz"
#   curl -sL -o grafana.tar.gz "https://dl.grafana.com/oss/release/grafana-13.2.0.linux-amd64.tar.gz"
#   tar xzf prometheus.tar.gz && tar xzf grafana.tar.gz && rm prometheus.tar.gz grafana.tar.gz
#
# Ports 9091/3001, not Prometheus/Grafana's defaults 9090/3000: this user
# already runs an unrelated Prometheus+Grafana pair (a different project)
# bound to the defaults — check with `ss -tln | grep -E ':9090|:3000'`
# before assuming the defaults are free on any given machine.

set -euo pipefail

PROMETHEUS_BIN="${PROMETHEUS_BIN:-$HOME/tools/prometheus-3.14.0.linux-amd64/prometheus}"
GRAFANA_HOMEPATH="${GRAFANA_HOMEPATH:-$HOME/tools/grafana-13.2.0}"
SERVELLM_TARGET="${SERVELLM_TARGET:-dgx-v100-01:18742}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DASHBOARD_DIR="$REPO_ROOT/dashboard"
RUNTIME_DIR="$DASHBOARD_DIR/.runtime"
mkdir -p "$RUNTIME_DIR/prometheus-data" "$RUNTIME_DIR/grafana-data" "$RUNTIME_DIR/provisioning/datasources" "$RUNTIME_DIR/provisioning/dashboards"

if ss -tln 2>/dev/null | grep -q ":9091 "; then
    echo "port 9091 already in use — is this already running? See scripts/observability_stop.sh" >&2
    exit 1
fi
if ss -tln 2>/dev/null | grep -q ":3001 "; then
    echo "port 3001 already in use — is this already running? See scripts/observability_stop.sh" >&2
    exit 1
fi

# Prometheus config: target is fixed (dgx-v100-01 is this project's one
# verified working serving node — see docs/ROADMAP.md), copied as-is.
sed "s#dgx-v100-01:18742#$SERVELLM_TARGET#" "$DASHBOARD_DIR/prometheus.local.yml" > "$RUNTIME_DIR/prometheus.yml"

# Grafana provisioning: substitute the absolute paths that can't be known
# until the repo is actually checked out somewhere.
sed "s#__DASHBOARD_DIR__#$DASHBOARD_DIR/grafana#" \
    "$DASHBOARD_DIR/grafana/provisioning/dashboards/servellm.yml" > "$RUNTIME_DIR/provisioning/dashboards/servellm.yml"
cp "$DASHBOARD_DIR/grafana/provisioning/datasources/servellm.yml" "$RUNTIME_DIR/provisioning/datasources/servellm.yml"
sed -e "s#__GRAFANA_DATA_DIR__#$RUNTIME_DIR/grafana-data#" \
    -e "s#__PROVISIONING_DIR__#$RUNTIME_DIR/provisioning#" \
    "$DASHBOARD_DIR/grafana.ini" > "$RUNTIME_DIR/grafana.ini"

nohup "$PROMETHEUS_BIN" \
    --config.file="$RUNTIME_DIR/prometheus.yml" \
    --storage.tsdb.path="$RUNTIME_DIR/prometheus-data" \
    --web.listen-address=127.0.0.1:9091 \
    > "$RUNTIME_DIR/prometheus.log" 2>&1 &
disown
echo $! > "$RUNTIME_DIR/prometheus.pid"

nohup "$GRAFANA_HOMEPATH/bin/grafana" server \
    --homepath "$GRAFANA_HOMEPATH" \
    --config "$RUNTIME_DIR/grafana.ini" \
    > "$RUNTIME_DIR/grafana.log" 2>&1 &
disown
echo $! > "$RUNTIME_DIR/grafana.pid"

echo "Prometheus starting (pid $(cat "$RUNTIME_DIR/prometheus.pid")), scraping $SERVELLM_TARGET"
echo "Grafana starting (pid $(cat "$RUNTIME_DIR/grafana.pid")), dashboard 'ServeLLM' auto-provisioned"
echo ""
echo "View from your own machine via SSH port-forward, e.g.:"
echo "  ssh -L 3001:localhost:3001 <you>@<login-node>"
echo "  then open http://localhost:3001  (login: admin / admin)"
echo ""
echo "Stop with: scripts/observability_stop.sh"
