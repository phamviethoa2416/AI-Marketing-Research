#!/usr/bin/env bash
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-masadmin}"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/multi-agent-system}"
MIN_RAM_GB=8

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
step()  { echo -e "\n${BLUE}▶${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
die()   { echo -e "${RED}✗${NC} $1" >&2; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────

step "Pre-flight checks"
[[ $EUID -eq 0 ]] || die "Must run as root or sudo"
. /etc/os-release
[[ "$ID" == "ubuntu" ]] || warn "Tested on Ubuntu; $ID may differ"

RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if (( RAM_GB < MIN_RAM_GB )); then
    warn "Only ${RAM_GB}GB RAM detected — recommended ≥${MIN_RAM_GB}GB for BGE-M3 models"
fi

DISK_GB=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')
(( DISK_GB >= 20 )) || warn "Low disk space: ${DISK_GB}GB (recommend ≥50GB)"
ok "Pre-flight done (RAM: ${RAM_GB}GB, Disk: ${DISK_GB}GB free)"

# ── System packages ───────────────────────────────────────────────────────────

step "Installing system packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release \
    git rsync unzip jq htop \
    ufw fail2ban
ok "System packages installed"

# ── Docker ────────────────────────────────────────────────────────────────────

step "Installing Docker Engine"
if command -v docker &>/dev/null; then
    ok "Docker already installed: $(docker --version)"
else
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
    ok "Docker installed: $(docker --version)"
fi

# ── Deploy user & permissions ─────────────────────────────────────────────────

step "Setting up deploy user: $DEPLOY_USER"
if ! id "$DEPLOY_USER" &>/dev/null; then
    useradd -m -s /bin/bash -G docker "$DEPLOY_USER"
    ok "User '$DEPLOY_USER' created"
else
    usermod -aG docker "$DEPLOY_USER"
    ok "User '$DEPLOY_USER' already exists"
fi

# ── SSH authorized keys for GitHub Actions ────────────────────────────────────

step "Setting up SSH for GitHub Actions deploy"
SSH_DIR="/home/$DEPLOY_USER/.ssh"
mkdir -p "$SSH_DIR"
touch "$SSH_DIR/authorized_keys"
chmod 700 "$SSH_DIR"
chmod 600 "$SSH_DIR/authorized_keys"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$SSH_DIR"

cat << 'EOF'

  ACTION REQUIRED:
  ─────────────────────────────────────────────────────────────
  Generate a deploy key pair on your LOCAL machine:

    ssh-keygen -t ed25519 -C "github-actions-mas" -f ~/.ssh/mas_deploy

  Then add the PUBLIC key to this server:
    cat ~/.ssh/mas_deploy.pub | sudo tee -a /home/masadmin/.ssh/authorized_keys

  And add the PRIVATE key to GitHub Secrets as: SERVER_SSH_KEY
  ─────────────────────────────────────────────────────────────
EOF

# ── Deploy directory ──────────────────────────────────────────────────────────

step "Creating deploy directory: $DEPLOY_PATH"
mkdir -p "$DEPLOY_PATH"
chown "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_PATH"
ok "Deploy path ready"

# ── Firewall ──────────────────────────────────────────────────────────────────

step "Configuring UFW firewall"
ufw --force reset > /dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8000/tcp comment 'MAS Gateway'
ufw allow 3000/tcp comment 'Grafana'
ufw allow 9090/tcp comment 'Prometheus (restrict to your IP in prod!)'
ufw --force enable
ok "Firewall configured"

# ── Swap (helps with ML model loading on RAM-limited servers) ─────────────────

step "Checking swap"
SWAP_GB=$(free -g | awk '/^Swap:/{print $2}')
if (( SWAP_GB < 4 )); then
    SWAP_FILE=/swapfile
    fallocate -l 8G "$SWAP_FILE" 2>/dev/null || dd if=/dev/zero of="$SWAP_FILE" bs=1M count=8192 status=none
    chmod 600 "$SWAP_FILE"
    mkswap "$SWAP_FILE" > /dev/null
    swapon "$SWAP_FILE"
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
    ok "8GB swap created"
else
    ok "Swap already configured: ${SWAP_GB}GB"
fi

# ── Docker daemon tuning ──────────────────────────────────────────────────────

step "Tuning Docker daemon"
cat > /etc/docker/daemon.json << 'DOCKERJSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
DOCKERJSON
systemctl reload docker
ok "Docker daemon configured"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════"
echo "  Bootstrap complete ✓"
echo "════════════════════════════════════════════"
echo ""
echo "  Deploy user : $DEPLOY_USER"
echo "  Deploy path : $DEPLOY_PATH"
echo ""
echo "  Next steps:"
echo "  1. Add deploy SSH public key (see above)"
echo "  2. Add GitHub Secrets:"
echo "     SERVER_HOST, SERVER_USER, SERVER_PORT"
echo "     SERVER_SSH_KEY, DEPLOY_PATH"
echo "     POSTGRES_PASSWORD, REDIS_PASSWORD"
echo "     QDRANT_API_KEY, TAVILY_API_KEY, SECRET_KEY"
echo "  3. Push to main → CI/CD triggers automatically"
echo ""