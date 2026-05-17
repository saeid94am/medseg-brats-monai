#!/usr/bin/env bash
# Download BraTS-GLI 2023 data via the Synapse CLI.
#
# Prerequisites:
#   1. Create a free account at https://www.synapse.org
#   2. Register for the BraTS 2023 challenge at https://www.synapse.org/#!Synapse:syn51156910
#   3. Install synapseclient: pip install synapseclient
#   4. Configure credentials: synapse login --remember-me
#
# Usage:
#   bash scripts/download_data.sh                    # downloads to data/BraTS2023_GLI/
#   bash scripts/download_data.sh /path/to/data      # custom output directory

set -euo pipefail

SYNAPSE_ID="syn51156910"          # BraTS 2023 GLI training data
OUTPUT_DIR="${1:-data/BraTS2023_GLI}"

echo "=== BraTS-GLI 2023 Download ==="
echo "Synapse entity : $SYNAPSE_ID"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check synapseclient is installed
if ! python -c "import synapseclient" &>/dev/null; then
    echo "ERROR: synapseclient not found. Run: pip install synapseclient"
    exit 1
fi

# Check credentials
if ! synapse whoami &>/dev/null; then
    echo "ERROR: Not logged in to Synapse."
    echo "Run: synapse login --remember-me"
    echo "Or set SYNAPSE_AUTH_TOKEN environment variable."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Downloading from Synapse (this may take a while — ~21 GB)..."
synapse get -r "$SYNAPSE_ID" --downloadLocation "$OUTPUT_DIR"

echo ""
echo "Download complete. Generating train/val splits..."
python src/medseg_brats/data/split_generator.py \
    --data_dir "$OUTPUT_DIR" \
    --val_frac 0.2 \
    --seed 42

echo ""
echo "Done. Files saved to $OUTPUT_DIR"
echo "Splits JSON saved to data/brats2023_splits.json"
