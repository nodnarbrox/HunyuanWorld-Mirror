#!/bin/bash
# One-shot setup of the m40-maxwell WorldMirror fork on the Tesla M40 box.
set -e
DEST=/opt/worldmirror
if ! mkdir -p "$DEST" 2>/dev/null; then
  DEST="$HOME/worldmirror"
  mkdir -p "$DEST"
fi
echo "DEST=$DEST"
cd "$DEST"
if [ ! -d .git ]; then
  git clone -q -b m40-maxwell https://github.com/nodnarbrox/HunyuanWorld-Mirror.git .
else
  git fetch -q origin m40-maxwell && git checkout -q m40-maxwell && git pull -q
fi
if [ ! -f venv/bin/python ]; then
  python3 -m venv venv
fi
. venv/bin/activate
pip -q install --upgrade pip
# last torch wheel line with sm_52 kernels
pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip -q install -r requirements_m40.txt
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cap", torch.cuda.get_device_capability())
print("bf16 supported:", torch.cuda.is_bf16_supported())
EOF
echo SETUP_OK
