#!/usr/bin/env bash
# e02 ESPnet track: "run.sh prepare" runs recipe stages 1 to 10 once; "run.sh <arm> [seed]" trains (11), decodes greedy (12) and scores
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
step=${1:?usage: run.sh prepare, or run.sh attention|grassmann|symmetric|no_mixer [seed]}
seed=${2:-0}
. "$root/vendor/espnet/tools/activate_python.sh"
cd "$root/vendor/espnet/egs2/librispeech_100/asr1"
# Options of the recipe's run.sh (v.202611), plus greedy CTC decoding of the averaged 10 best checkpoints by validation CER
recipe=(--python "$here/python" --lang en --ngpu 1 --nj 16 --gpu_inference true --inference_nj 2 --nbpe 5000 --max_wav_duration 30
        --speed_perturb_factors "0.9 1.0 1.1" --audio_format flac.ark --feats_type raw --use_lm false
        --train_set train_clean_100 --valid_set dev --test_sets "test_clean test_other dev_clean dev_other"
        --lm_train_text data/train_clean_100/text --bpe_train_text data/train_clean_100/text
        --inference_config conf/tuning/decode_ctc_bs1.yaml --inference_tag greedy --inference_asr_model valid.cer_ctc.ave.pth)
if [ "$step" = prepare ]; then
  ./asr.sh "${recipe[@]}" --asr_config "$here/conf/attention.yaml" --stage 1 --stop_stage 10
  exit 0
fi
tag=${step}_s${seed}
./asr.sh "${recipe[@]}" --asr_config "$here/conf/$step.yaml" --asr_tag "$tag" --asr_args "--seed $seed" --stage 11 --stop_stage 12
out=$root/reports/e02_espnet/$tag
mkdir -p "$out"
cp "exp/asr_$tag/config.yaml" "$out/"
for dset in dev_clean dev_other test_clean test_other; do
  cp "exp/asr_$tag/greedy/$dset/text" "$out/$dset.hyp"
  python3 -m grassmann_speech.wer "data/$dset/text" "$out/$dset.hyp" > "$out/$dset.json"
done
python3 -c "import json, pathlib; from grassmann_speech.env import environment; print(json.dumps(environment(pathlib.Path('$root')), indent=1))" > "$out/env.json"
