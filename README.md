# TC-SAE-RAE

`TC-SAE`와 `SAE-RAE` 실험 코드를 한 저장소 안에서 역할별로 분리해 정리한 작업 트리입니다.

## 구조

```text
TC-SAE-RAE/
  vendor/
    rae_src/
  src/
    tc_sae/
      train.py
      discriminator.py
    sae_rae/
      train.py
      cache_latents.py
      infer.py
      eval.py
      eval_fid.py
      make_cfg_grids.py
      make_pair_grids_from_ckpt.py
      visualize_latents.py
      conditioning.py
      script_utils.py
    sae_local/
      model.py
      loss.py
  configs/
    tc_sae/
    sae_rae/
    vendor/
```

## 역할

- `src/tc_sae/`: TC regularization 기반 SAE 학습 코드
- `src/sae_rae/`: SAE-conditioned RAE 학습, 캐시, 추론, 평가 코드
- `src/sae_local/`: SAE 코어 구현
- `vendor/rae_src/`: 원본 RAE 의존 코드
- `configs/tc_sae/`: TC-SAE 실험 설정
- `configs/sae_rae/`: SAE-RAE 실험 설정

## 예시 실행

- `python src/tc_sae/train.py --config configs/tc_sae/ffhq256_sae_tc_preact_v1.yaml`
- `python src/sae_rae/train.py --config configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS.yaml ...`
- `python src/sae_rae/cache_latents.py --config ...`
- `python src/sae_rae/infer.py --config configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS.yaml ...`
- `python src/sae_rae/label_latents.py --config ... --data-path /path/to/val --output-dir results/latent_labels`
  - `--qwen-model`을 주면 Qwen-VL 라벨링은 `vLLM`으로 실행됩니다.

## 제외한 항목

- 실험 로그와 결과물
- 대용량 체크포인트와 캐시
- 일회성 분석 스크립트 다수

이관 기준은 [migration.md](/scratch/x3411a10/TC-SAE-RAE/docs/migration.md:1)에 정리되어 있습니다.
