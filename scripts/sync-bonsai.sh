#!/bin/sh
# Copies the Bonsai runtime (shared Rust, Metal and Vulkan layers, kernels) from Glimpse-Lab/g1/rust into
# src/bonsai/runtime and reapplies our patch.
# Usage: scripts/sync-bonsai.sh [path to Glimpse-Lab/g1/rust]
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
src=${1:-"$repo/../Glimpse-Lab/g1/rust"}
dst="$repo/src/bonsai/runtime"

for f in gpu gpu_vk lora model tokenizer vision weights; do
    sed 's/crate::/super::/g' "$src/src/$f.rs" > "$dst/$f.rs"
done
sed 's/crate::/super::/g' "$src/src/lib.rs" > "$dst/mod.rs"
cp "$src/src/kernels.metal" "$dst/kernels.metal"
rm -rf "$dst/kernels"
cp -R "$src/src/kernels" "$dst/kernels"

patch -s -p1 -d "$repo" < "$repo/scripts/bonsai-runtime.patch"
echo "synced from $src"
