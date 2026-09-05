#!/bin/bash
# Phase 14: lints and renders the Helm chart (deploy/helm/servellm), then
# validates the rendered YAML against real Kubernetes API schemas — the
# actual verification available in a dev environment with no Docker daemon
# and no Kubernetes cluster (this cluster's login node; see
# docs/ROADMAP.md's Phase 14 section). Doesn't require either: `helm lint`
# and `helm template` only read the chart's own files, and `kubeconform`
# validates YAML shape against a downloaded schema catalog, no live
# apiserver involved.
#
# One-time setup (not run by this script — do this once):
#   mkdir -p ~/tools && cd ~/tools
#   curl -sL -o helm.tar.gz "https://get.helm.sh/helm-v4.2.4-linux-amd64.tar.gz"
#   tar xzf helm.tar.gz && mv linux-amd64 helm-v4.2.4-linux-amd64 && rm helm.tar.gz
#   mkdir -p kubeconform && cd kubeconform
#   curl -sL -o kc.tar.gz "https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz"
#   tar xzf kc.tar.gz && rm kc.tar.gz

set -euo pipefail

HELM_BIN="${HELM_BIN:-$HOME/tools/helm-v4.2.4-linux-amd64/helm}"
KUBECONFORM_BIN="${KUBECONFORM_BIN:-$HOME/tools/kubeconform/kubeconform}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHART_DIR="$REPO_ROOT/deploy/helm/servellm"

echo "=== helm lint ==="
"$HELM_BIN" lint "$CHART_DIR"

echo ""
echo "=== helm template (default values) + kubeconform ==="
"$HELM_BIN" template default-check "$CHART_DIR" | "$KUBECONFORM_BIN" -strict -summary

echo ""
echo "=== helm template (autoscaling + serviceMonitor enabled) + kubeconform ==="
"$HELM_BIN" template full-check "$CHART_DIR" \
    --set autoscaling.enabled=true \
    --set serviceMonitor.enabled=true \
  | "$KUBECONFORM_BIN" -strict -summary \
    -schema-location default \
    -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'

echo ""
echo "=== rendered models ConfigMap parses through the real registry code ==="
"$HELM_BIN" template models-check "$CHART_DIR" --show-only templates/configmap.yaml \
    | python3 -c "
import sys, re
content = sys.stdin.read()
match = re.search(r'data:\n  models\.yaml: \|\n((?:    .*\n?)+)', content)
lines = match.group(1).splitlines()
dedented = '\n'.join(l[4:] for l in lines)
open('/tmp/servellm-helm-check-models.yaml', 'w').write(dedented)
"
PYTHONPATH="$REPO_ROOT" python3 -c "
from backend.router.registry import load_model_specs
specs = load_model_specs('/tmp/servellm-helm-check-models.yaml')
print(f'{len(specs)} model(s) parsed correctly:', [s.served_model_id for s in specs])
"
rm -f /tmp/servellm-helm-check-models.yaml

echo ""
echo "All checks passed."
