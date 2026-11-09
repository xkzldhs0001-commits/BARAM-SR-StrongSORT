#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/StrongSORT"
EXPECTED='ee995076da5083e28d0da1f885297df62705ebd7'
if [[ ! -d "$SRC/.git" ]];then echo '[FAIL] Official Git repository is missing' >&2;exit 2;fi
REV="$(git -C "$SRC" rev-parse HEAD)"
echo "[AUDIT] dyhBUPT/StrongSORT HEAD: $REV"
if [[ "$REV" != "$EXPECTED" ]];then echo "[FAIL] Expected $EXPECTED" >&2;exit 3;fi
if ! git -C "$SRC" diff --quiet HEAD -- deep_sort strong_sort.py deep_sort_app.py AFLink GSI.py opts.py;then
 echo '[FAIL] Original tracking source differs from pinned author commit!' >&2
 git -C "$SRC" status --short >&2
 exit 4
fi
echo '[PASS] Original StrongSORT and StrongSORT++ source files match pinned author commit.'
echo '[NOTE] runtime Python compat adapters and external detector/extractor are BARAM code, not original paper pipeline.'
