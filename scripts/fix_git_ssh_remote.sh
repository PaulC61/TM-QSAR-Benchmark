#!/usr/bin/env bash
# Diagnose + fix "GitHub keeps asking for username/password" on a server
# that already has a working SSH key (i.e. `ssh -T git@github.com` prints
# "Hi <user>! You've successfully authenticated...").
#
# Usage: run from inside the repo on the server, e.g.:
#   cd ~/tm-descriptor-benchmark && bash scripts/fix_git_ssh_remote.sh
#
# It checks, in order, the four things that can still force HTTPS even
# though SSH auth itself works, and rewrites them to SSH:
#   1. The repo's own `origin` remote
#   2. Any submodule remotes
#   3. Global `url.<ssh>.insteadOf <https>` rewrites that undo #1/#2
#   4. A stale HTTPS credential helper entry that git tries first
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Not inside a git repo. cd into the repo first." >&2
    exit 1
fi

echo "== 1. SSH auth sanity check =="
ssh -T git@github.com -o BatchMode=yes -o ConnectTimeout=5 2>&1 || true
echo

echo "== 2. Current 'origin' remote =="
git remote -v
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || true)
if [[ "$ORIGIN_URL" == https://github.com/* ]]; then
    SSH_URL="git@github.com:${ORIGIN_URL#https://github.com/}"
    SSH_URL="${SSH_URL%.git}.git"
    echo "  -> origin is HTTPS, rewriting to: $SSH_URL"
    git remote set-url origin "$SSH_URL"
    git remote -v
else
    echo "  -> origin is already SSH (or not github.com), leaving as-is."
fi
echo

echo "== 3. Submodule remotes =="
if [ -f .gitmodules ]; then
    git submodule foreach 'echo "--- $name ---"; git remote -v' 2>&1 || true
    # Rewrite any HTTPS submodule remotes to SSH too.
    git submodule foreach '
        url=$(git remote get-url origin 2>/dev/null || true)
        case "$url" in
            https://github.com/*)
                ssh_url="git@github.com:${url#https://github.com/}"
                case "$ssh_url" in *.git) ;; *) ssh_url="${ssh_url}.git" ;; esac
                echo "  -> $name origin is HTTPS, rewriting to: $ssh_url"
                git remote set-url origin "$ssh_url"
                ;;
        esac
    ' 2>&1 || true
else
    echo "  -> no .gitmodules, skipping."
fi
echo

echo "== 4. Global 'url.insteadOf' rewrites (can silently force HTTPS) =="
INSTEADOF=$(git config --global --get-regexp 'url\..*insteadof' 2>/dev/null || true)
if [ -n "$INSTEADOF" ]; then
    echo "$INSTEADOF"
    echo "  -> If any of the above maps ssh://... or git@github.com to"
    echo "     https://github.com, remove it with:"
    echo "       git config --global --unset-all url.<name>.insteadOf"
else
    echo "  -> none found."
fi
echo

echo "== 5. Credential helper (can pop a prompt even for SSH remotes if" \
     "something re-triggers HTTPS) =="
git config --show-origin --get-all credential.helper 2>&1 || echo "  -> none configured."
echo

echo "== Done. Try: git pull =="
