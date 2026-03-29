#!/usr/bin/env bash

set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
MAX_RETRIES="${MAX_RETRIES:-12}"
SLEEP="${SLEEP:-5}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
info() { echo -e "${YELLOW}→${NC} $1"; }

info "Waiting for gateway at $GATEWAY_URL ..."
for i in $(seq 1 "$MAX_RETRIES"); do
    if curl -sf "$GATEWAY_URL/health" > /dev/null 2>&1; then
        pass "Gateway is up"
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        fail "Gateway did not come up after $((MAX_RETRIES * SLEEP))s"
        exit 1
    fi
    echo "  Retry $i/$MAX_RETRIES..."
    sleep "$SLEEP"
done

HEALTH=$(curl -sf "$GATEWAY_URL/health")
OVERALL=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")

echo ""
echo "Overall: $OVERALL"
echo ""

echo "$HEALTH" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for s in d.get('services', []):
    icon = '✓' if s['status'] == 'ok' else ('⚠' if s['status'] == 'degraded' else '✗')
    lat  = f\"{s.get('latency_ms', '?')}ms\"
    print(f\"  {icon}  {s['name']:15} {s['status']:10} {lat}\")
    if s['status'] == 'down':
        detail = s.get('detail', '')
        print(f\"      Detail: {detail}\")
"

echo ""
if [ "$OVERALL" = "ok" ]; then
    pass "All services healthy — deploy verified ✓"
    exit 0
elif [ "$OVERALL" = "degraded" ]; then
    fail "Some services degraded — check logs"
    exit 1
else
    fail "System unhealthy"
    exit 1
fi