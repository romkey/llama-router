#!/bin/sh
# Prints the CHANGELOG.md section for a version, for use as release notes.
# Fails if the section is missing or empty, which stops a release that
# forgot to update the changelog.
#
#   scripts/changelog-section.sh 1.2.0
set -eu
version=${1#v}
changelog=${CHANGELOG:-CHANGELOG.md}
notes=$(awk -v v="$version" '
  /^## \[/ { if (found) exit; if (index($0, "## [" v "]") == 1) { found = 1; next } }
  found { print }
' "$changelog")
if [ -z "$(printf '%s' "$notes" | tr -d '[:space:]')" ]; then
  echo "$changelog has no entries under ## [$version]" >&2
  exit 1
fi
printf '%s\n' "$notes"
