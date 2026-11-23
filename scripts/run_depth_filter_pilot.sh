#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mode="${1:---depth-pilot}"
if [[ "$#" -gt 1 || ( "$mode" != --depth-pilot && "$mode" != --architecture ) ]]; then
    printf 'Usage: bash scripts/run_depth_filter_pilot.sh [--depth-pilot|--architecture]\n' >&2
    exit 2
fi
common=(
    --method tikh --filter-steps 5
    --heads 8 --context 32
    --batch 8 --steps 500 --seed 17 --device cpu
    --subset sample-10BT
    --revision 9bb295ddab0e05d785b879661af7260fed5140fc
    --shuffle-buffer 1000 --eval-every 100 --eval-batches 10
    --capture 1 10 50 100 250 500
)

if [[ "$mode" == --architecture ]]; then
    for configuration in 64:6 128:6 88:4; do
        output="results/architecture_spectral_grid/w${configuration%:*}_d${configuration#*:}_tikh_lam0_seed17"
        if [[ -e "$output" ]]; then
            printf 'Refusing to overwrite %s\n' "$output" >&2
            exit 1
        fi
    done
    for configuration in 64:6 128:6 88:4; do
        output="results/architecture_spectral_grid/w${configuration%:*}_d${configuration#*:}_tikh_lam0_seed17"
        python3 scripts/train.py "${common[@]}" \
            --width "${configuration%:*}" --layers "${configuration#*:}" --tikh-lambda 0 \
            --out "$output" &
        worker=$!
        trap 'kill -TERM "$worker" 2>/dev/null || true; wait "$worker" 2>/dev/null || true; exit 130' INT TERM
        completed_wait=0
        cleaned_up=0
        while kill -0 "$worker" 2>/dev/null; do
            if [[ -f "$output/results.json" ]]; then
                completed_wait=$((completed_wait + 1))
                if [[ "$completed_wait" -ge 20 ]]; then
                    python3 -c 'import json, math, pathlib, sys; root = pathlib.Path(sys.argv[1]); data = json.loads((root / "results.json").read_text()); assert len(data["losses"]) == 500 and all(math.isfinite(loss) for loss in data["losses"]); assert data["validation_losses"][-1]["step"] == 500; assert all((root / name).is_file() for name in ("checkpoint.pt", "momenta.pt"))' "$output"
                    printf 'Results saved; stopping worker waiting on streamer shutdown: %s\n' "$output"
                    if kill -TERM "$worker" 2>/dev/null; then
                        cleaned_up=1
                    fi
                    break
                fi
            fi
            sleep 1
        done
        if [[ "$cleaned_up" -eq 1 ]]; then
            status=0
            wait "$worker" || status=$?
            if [[ "$status" -ne 0 && "$status" -ne 143 ]]; then
                exit "$status"
            fi
        else
            wait "$worker"
        fi
        trap - INT TERM
    done
    exit 0
fi

python3 scripts/train.py "${common[@]}" --layers 8 --width 88 --tikh-lambda 0 \
    --out results/depth_filter_pilot/w88_d8_tikh_lam0_seed17
python3 scripts/train.py "${common[@]}" --layers 8 --width 88 --tikh-lambda 6.1352038e-5 \
    --out results/depth_filter_pilot/w88_d8_tikh_lam6p14e-5_seed17
