#!/usr/bin/env bash
set -u

GPU="${1:?usage: $0 GPU RUN CKPT OUT_ROOT [SEED START_INDEX END_INDEX]}"
RUN="${2:?missing run}"
CKPT="${3:?missing checkpoint}"
OUT_ROOT="${4:?missing output root}"
SEED="${5:-20260734}"
START_INDEX="${6:-0}"
END_INDEX="${7:-5}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
EVAL="${ROOT}/legged_gym/scripts/run_physical_jump_checkpoint_eval_remote.sh"

names=(
    formal_opt
    final_combined
    broad_random_success
    broad_random_zero
    extreme_low
    extreme_high
)
thigh=(1.0722073939 1.0341343050 0.8392214284 1.1511763841 0.8 1.2)
shank=(1.0502235759 1.0049821020 0.8526662940 0.8894574749 0.8 1.2)
abad=(0.7090862835 1.4980984860 0.7219795756 1.2724573530 0.6 1.6)
hip=(0.6079164132 0.8170974290 1.1901934785 0.9076988220 0.6 1.6)
knee=(1.1493373415 1.5785909830 1.4551206475 1.4768658223 0.6 1.6)
wheel=(0.8220149711 1.5560149750 0.6528457721 1.4986159119 0.6 1.6)

for index in "${!names[@]}"; do
    if (( index < START_INDEX || index > END_INDEX )); then
        continue
    fi
    env \
        JUMPFWD_EVAL_ABAD="${abad[index]}" \
        JUMPFWD_EVAL_HIP="${hip[index]}" \
        JUMPFWD_EVAL_KNEE="${knee[index]}" \
        JUMPFWD_EVAL_WHEEL="${wheel[index]}" \
        bash "${EVAL}" \
        "${GPU}" \
        "${RUN}" \
        "${CKPT}" \
        "${OUT_ROOT}/${names[index]}" \
        "${SEED}" \
        "${thigh[index]}" \
        "${shank[index]}" \
        1.0 || exit $?
done
