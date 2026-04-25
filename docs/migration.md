# Migration Notes

## 2026-04-24 구조 재정리

실험 코드를 아래 기준으로 다시 배치했습니다.

- `src/tc_sae/`: TC regularization 기반 SAE 학습 코드
- `src/sae_rae/`: SAE-conditioned RAE 학습/추론/평가 코드
- `src/sae_local/`: 재사용되는 SAE 코어 구현
- `vendor/rae_src/`: 원본 RAE vendor 코드 유지

설정 파일도 역할 기준으로 분리했습니다.

- `configs/tc_sae/`
- `configs/sae_rae/`
- `configs/vendor/`

## 이전 경로에서 바뀐 항목

- `scripts/train_sae_tc.py` -> `src/tc_sae/train.py`
- `src/discriminator.py` -> `src/tc_sae/discriminator.py`
- `src/train_sae_cond.py` -> `src/sae_rae/train.py`
- `src/cache_sae_rae_latents.py` -> `src/sae_rae/cache_latents.py`
- `src/infer.py` -> `src/sae_rae/infer.py`
- `src/eval_fid.py` -> `src/sae_rae/eval_fid.py`
- `src/eval_base_val_loss.py` -> `src/sae_rae/eval.py`
- `src/make_cfg_grids.py` -> `src/sae_rae/make_cfg_grids.py`
- `src/make_pair_grids_from_ckpt.py` -> `src/sae_rae/make_pair_grids_from_ckpt.py`
- `src/visualize_sae_latents.py` -> `src/sae_rae/visualize_latents.py`
- `configs/tc_sar/*` -> `configs/tc_sae/*`
- `configs/stage2/training/*` -> `configs/sae_rae/*`

## 의도

- 실험 축별로 파일 위치가 바로 보이게 하기
- `vendor` 코드와 우리 코드를 물리적으로 분리하기
- 루트 `src/`에 흩어져 있던 실행 스크립트를 줄이기
- 경로 의존성을 줄이고 유지보수 부담을 낮추기
