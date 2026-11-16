#!/usr/bin/env bash
# e02 ESPnet track setup: pinned recipe checkout in vendor/espnet, its own uv env, LibriSpeech shared with the own trainer in data/librispeech
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
espnet=$root/vendor/espnet
if [ ! -d "$espnet" ]; then
  git clone --depth 1 --branch v.202611 --filter=blob:none --sparse https://github.com/espnet/espnet "$espnet"
fi
# librispeech_100/asr1/local and TEMPLATE/asr1/{utils,scripts,pyscripts} symlink into these two
git -C "$espnet" sparse-checkout set egs2/TEMPLATE egs2/librispeech_100 egs2/librispeech/asr1/local utils tools
uv sync --project "$here" --locked
# path.sh sources this file when present: the recipe then runs in the uv env and sees this repo's layer code
cat > "$espnet/tools/activate_python.sh" <<ACTIVATE
. "$here/.venv/bin/activate"
export PYTHONPATH="$root/python/src:$here\${PYTHONPATH:+:\$PYTHONPATH}"
ACTIVATE
mkdir -p "$root/data/librispeech"
ln -sfn "$root/data/librispeech" "$espnet/egs2/librispeech_100/asr1/downloads"
for tool in sox flac; do
  command -v "$tool" >/dev/null || echo "missing $tool: the recipe's data prep and speed perturbation call it (apt-get install sox flac)"
done
