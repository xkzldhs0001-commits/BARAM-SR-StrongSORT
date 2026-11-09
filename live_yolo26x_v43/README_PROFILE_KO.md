# BARAM V4.3 Runtime Bottleneck Profiler — VS Code

기존 **StrongSORT + YOLO26x (baseline)** 와 **SR-StrongSORT V4.3 Selective + YOLO26x (OURS)** 의
단계별 지연시간과 검출 인원 수에 따른 지연시간을 분석합니다. V4.3 / StrongSORT 추적 코드는 덮어쓰지 않습니다.

## 1. 적용

압축 파일을 Ubuntu `~/Downloads`에 저장한 뒤:

```bash
cd ~/Downloads/BARAM_LIVE_YOLO26X_OCCLUSION_V4_1
unzip -o ~/Downloads/BARAM_RUNTIME_PROFILER_VSCODE.zip -d .
code BARAM_Runtime_Profiling.code-workspace
```

`Terminal → Run Task → BARAM: existing A/B -> bottleneck charts (no GPU)` 를 클릭하면 됩니다.

### 가장 빠른 방법: 예전 45분짜리 실험 결과 그대로 분석

```bash
cd ~/Downloads/BARAM_LIVE_YOLO26X_OCCLUSION_V4_1
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python run_baram_runtime_profile.py \
  --from-ab ~/Downloads/BARAM_SR_StrongSORT/outputs/BARAM_YOLO26X_STRONGSORT_AB/20261106_173209
```

이미 생성된 `runs/MOT17/.../baseline/.../frames.csv` 및
`runs/MOT17/.../v43_selective/.../frames.csv` (MOT20 동일)를 분석합니다.
**GPU 재추론 없음**, TrackEval 재실행 없음, 발표용 실제 측정값만 계산합니다.
분석 대상은 기본적으로 해당 실험의 모든 MOT17/MOT20 시퀀스입니다.

특정 두 시퀀스만 보려면:

```bash
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python run_baram_runtime_profile.py \
  --from-ab ~/Downloads/BARAM_SR_StrongSORT/outputs/BARAM_YOLO26X_STRONGSORT_AB/20261106_173209 \
  --only MOT17-10-FRCNN MOT20-02
```

### 더 정확하게 인원수 구간 측정: GPU 재실행 (선택 사항)

기존 `frames.csv`의 `detections`에는 **Occlusion Rescue의 저신뢰 후보도 포함**됩니다.
아래 새 측정 워커는 `high_conf_detections`를 별도 기록하므로 밀집도 비교의 설명이 더 명확합니다.
기본적으로 각 시퀀스 처음 600프레임을 처리합니다.

```bash
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python run_baram_runtime_profile.py \
  --measure --max-frames 600 \
  --only MOT17-10-FRCNN MOT20-02
```

전체 프레임으로 검사하려면 `--max-frames 0`, 실행 순서 영향을 확인하려면 `--repeats 2` 추가:

```bash
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python run_baram_runtime_profile.py \
  --measure --max-frames 0 --repeats 2 \
  --only MOT17-10-FRCNN MOT20-02
```

위 실행에는 NVIDIA CUDA가 정상 작동해야 하며 `YOLO26x`와 `OSNet`을 다시 실행하므로 시간이 걸립니다.
**동일 프레임, conf 0.40, detector floor 0.15, imgsz 640, no-render, no-display**로 두 모델 측정.
새 측정의 `--repeats 2`는 모델 실행 순서를 교차하고, 두 반복의 프레임을 집계합니다.

## 2. 결과물

일반 경로: `~/Downloads/BARAM_SR_StrongSORT/outputs/BARAM_RUNTIME_PROFILING/YYYYMMDD_HHMMSS/`

- `PROFILE_REPORT.md`: 데이터셋별 측정 보고서 (재실행 시 `report/` 하위)
- `stage_latency.csv`: YOLO26x / OSNet / Prediction & Gating / Data Association / ID Management / Other의 평균 지연시간
- `crowd_latency.csv`: 검출 박스 수 `0, 1~5, 6~15, 16+`별 평균 / P95 지연시간, 샘플 프레임 수
- `runtime_summary.csv`: end-to-end 평균 ms, P50/P95, FPS (프로파일링 기준), Re-ID crop 수
- `per_sequence_latency.csv`: 시퀀스별 세부 수치
- `measurement_protocol.json`: 출처·프레임 선택·해석 주의 사항
- `MOT17_stage_latency.png`, `MOT20_stage_latency.png`
- `MOT17_crowd_latency.png`, `MOT20_crowd_latency.png`
- `overleaf_values_MOT17.tex`, `overleaf_values_MOT20.tex`: 기존 Overleaf 그래프 0값과 교체할 수치 매크로

결과 폴더 확인:

```bash
xdg-open ~/Downloads/BARAM_SR_StrongSORT/outputs/BARAM_RUNTIME_PROFILING/
```

## 3. 실제 의미와 한계

- **YOLO26x Detection**: 객체 검출과 점수 필터.
- **OSNet Re-ID**: 선택된 사람 크롭 생성 및 특징 추출. CUDA 동기화 포함.
- **Prediction & Gating**: 트랙 예측, 선택적 특징 추출 후보 판단, 약한 박스 복구 후보 선별, ECC.
- **Data Association**: 검출객체 구성 및 tracker.update() (Hungarian 등 포함).
- **ID Management**: 표시 ID bank 복원·검증. baseline에서는 0에 가까울 수 있음.
- **I/O & Other**: `wall_ms`에서 위 처리 단계 합계를 뺀 잔여시간. 직접 별도로 측정한 CUDA 커널은 아닙니다.

**P95**는 지연시간 분포의 95백분위수입니다. 특정 프레임의 밀집도는 실제 GT 인원 수가 아닌
검출 박스 수로 분류한 *proxy*입니다. 오래된 A/B 데이터는 low-confidence rescue 후보가 포함됩니다.
낮은 신뢰도의 후보 포함 여부는 모델 간 달라질 수 있으므로 같은 데이터셋의 1:1 인원수 측정이라고
표현하면 안 됩니다. 재측정 시 신규 컬럼 `high_conf_detections`로 표시합니다.

Stage timer는 PyTorch CUDA 동기화를 포함하며, 샘플별 타이밍 기록 자체의 오버헤드가 있습니다.
그래서 이 결과의 FPS는 기존 HOTA용 비교표의 FPS와 다를 수 있습니다.
평가하는 것은 병목 **분포**이며, 총 FPS와의 인과관계가 엄격히 입증됐다는 주장은 피하세요.
여기서는 HOTA / MOTA / IDSW를 재평가하지 않습니다.

### Overleaf 표에 반영

이전에 작성한 `BARAM_Runtime_Bottleneck_Overleaf.tex` 상단의
`\\profilingreadyfalse`, `\\baseStage`, `\\oursStage`, `\\baseCrowd`, `\\oursCrowd`,
`\\stageAxisMax`, `\\crowdAxisMax` 선언을 출력 파일 `overleaf_values_MOT17.tex` 또는
`overleaf_values_MOT20.tex` 안의 동일 매크로들로 **교체**하면 실측 그래프가 나옵니다.
기존 Overleaf 그래프는 `Prediction & Gating` + `Data Association`을
하나의 `Gating + Matching` 칸으로 합산해 표시합니다.

## 4. 테스트

```bash
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python -m unittest -v test_profile_analysis
```

단위 테스트는 분석 로직/가짜 입력 확인용일 뿐 NVIDIA GPU 추론 성공의 증거는 아닙니다.
