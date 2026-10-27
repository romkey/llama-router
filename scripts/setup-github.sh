#!/bin/sh
# GitHub settings every repo following python-ai-rules uses. Safe to
# re-run. Needs gh, logged in as a repo admin.
#
#   scripts/setup-github.sh [owner/repo]    # defaults to the current repo
#
# - main is protected by a ruleset: changes only through pull requests,
#   squash-merged, with the "CI passed" and "Lint" checks green and the
#   branch up to date; no force-pushes or deletion.
# - Secret scanning with push protection, generic-secret patterns and
#   validity checks, Dependabot alerts (Renovate reads them), private
#   vulnerability reporting, and CodeQL default setup.
# - Workflow tokens are read-only unless a job asks for more, and workflows
#   may open PRs (the release workflow does).
# - Merged branches are deleted.
#
# Some security features need GitHub Advanced Security on private repos;
# those steps warn and carry on.
set -eu
repo=${1:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}
echo "Configuring $repo"

try() {
  what=$1
  shift
  if "$@" >/dev/null; then echo "  ok: $what"; else echo "  FAILED: $what" >&2; fi
}

try "squash merge only, delete merged branches" \
  gh api -X PATCH "repos/$repo" \
  -F allow_squash_merge=true -F allow_merge_commit=false -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true -F allow_update_branch=true \
  -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=PR_BODY

try "secret scanning and push protection" \
  gh api -X PATCH "repos/$repo" --input - <<'EOF'
{
  "security_and_analysis": {
    "secret_scanning": {"status": "enabled"},
    "secret_scanning_push_protection": {"status": "enabled"}
  }
}
EOF

# Generic secrets (passwords, connection strings) and checking whether found
# tokens are still live. These need GitHub Secret Protection; without it
# GitHub accepts the request but leaves them off, so check what stuck.
gh api -X PATCH "repos/$repo" --input - >/dev/null 2>&1 <<'EOF' || true
{
  "security_and_analysis": {
    "secret_scanning_non_provider_patterns": {"status": "enabled"},
    "secret_scanning_validity_checks": {"status": "enabled"}
  }
}
EOF
for feature in secret_scanning_non_provider_patterns secret_scanning_validity_checks; do
  status=$(gh api "repos/$repo" --jq ".security_and_analysis.$feature.status // \"unavailable\"")
  if [ "$status" = enabled ]; then
    echo "  ok: $feature"
  else
    echo "  skipped: $feature is $status (needs GitHub Secret Protection)"
  fi
done

try "Dependabot alerts" gh api -X PUT "repos/$repo/vulnerability-alerts"
try "private vulnerability reporting" gh api -X PUT "repos/$repo/private-vulnerability-reporting"
try "CodeQL default setup" gh api -X PATCH "repos/$repo/code-scanning/default-setup" -f state=configured

try "read-only workflow tokens; workflows may open PRs" \
  gh api -X PUT "repos/$repo/actions/permissions/workflow" \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=true

# 15368 is the GitHub Actions app, so only Actions can satisfy the checks.
ruleset=$(cat <<'EOF'
{
  "name": "Protect main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "bypass_actors": [],
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "allowed_merge_methods": ["squash"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          {"context": "CI passed", "integration_id": 15368},
          {"context": "Lint", "integration_id": 15368}
        ]
      }
    }
  ]
}
EOF
)
id=$(gh api "repos/$repo/rulesets" --jq '.[] | select(.name == "Protect main") | .id')
if [ -n "$id" ]; then
  echo "$ruleset" | try "update ruleset 'Protect main'" gh api -X PUT "repos/$repo/rulesets/$id" --input -
else
  echo "$ruleset" | try "create ruleset 'Protect main'" gh api -X POST "repos/$repo/rulesets" --input -
fi
