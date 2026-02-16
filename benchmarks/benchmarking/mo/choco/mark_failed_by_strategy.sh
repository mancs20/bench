#!/usr/bin/env bash
set -euo pipefail

JOBLOG="./jobs.log"
ALLOWED="./all-front-generators"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <strategy>"
  exit 1
fi

STRAT="$1"

if [ ! -f "$JOBLOG" ]; then
  echo "Error: missing $JOBLOG (run from the folder that contains it)"
  exit 1
fi

if [ ! -f "$ALLOWED" ]; then
  echo "Error: missing $ALLOWED (run from the folder that contains it)"
  exit 1
fi

# Safety: strategy must be exactly one line in the allowed list
if ! grep -Fxq "$STRAT" "$ALLOWED"; then
  echo "Error: strategy '$STRAT' not found in $ALLOWED"
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# TSV joblog:
# Seq Host Starttime JobRuntime Send Receive Exitval Signal Command
awk -v strat="$STRAT" 'BEGIN { FS=OFS="\t" }
NR==1 { print; next }
{
  # match strategy as a whole word inside Command column
  if ($9 ~ ("(^|[^[:alnum:]_])" strat "([^[:alnum:]_]|$)")) {
    if ($7 == 0) $7 = 1
  }
  print
}' "$JOBLOG" > "$TMP"

mv "$TMP" "$JOBLOG"

echo "Done: marked Exitval=1 for strategy '$STRAT' in $JOBLOG."
