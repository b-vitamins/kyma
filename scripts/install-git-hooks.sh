#!/usr/bin/env bash
set -euo pipefail

HOOK_PATH=".git/hooks/pre-commit"

cat > "${HOOK_PATH}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

bash scripts/pre-commit.sh
EOF

chmod +x "${HOOK_PATH}"
echo "Installed pre-commit hook at ${HOOK_PATH}"
