# BARAM V4.3 vs 일반 StrongSORT — YOLO26x 동일 조건 MOT17/20 비교

**목표:** 두 방식에서 검출기(YOLO26x), 가중치, 해상도, 입력 프레임, YOLO 검출 하한을 동일하게 유지하고 추적·ID 복원 알고리즘만 비교합니다.

| 모델 | 추적 구성 |
|---|---|
| `baseline` | 공식 StrongSORT `Tracker` 코어 + OSNet + YOLO26x (현재 프로젝트의 baseline 경로) |
| `v43_full` | SR 추적기 + V4.3 ID Memory / Relink / Occlusion Rescue / Verification + OSNet FULL + YOLO26x |
| `v43_selective` (선택) | 위 V4.3에 선택적 OSNet 특징 추출을 적용한 경량 모드 |

**주의:** `baseline`은 동일 프로젝트가 호출하는 원본 StrongSORT 추적기 코어를 사용한 비교 모델입니다. 정식 StrongSORT++의 AFLink/GSI 후처리는 적용하지 않습니다. 이는 논문에 보고된 YOLOX-X + StrongSORT 점수를 복제하는 실험이 아니라, **YOLO26x 통일 조건에서 비교하는 새로운 실험**입니다.

## 1. 다운로드한 ZIP 적용

1. ZIP을 Ubuntu `~/Downloads`에 저장합니다.
2. 아래 명령어로 설치합니다. 기존 `live.py`, `live_mot17.py`, `identity_memory.py`, `run.sh`는 **수정·삭제하지 않습니다**.

```bash
cd ~/Downloads/BARAM_LIVE_YOLO26X_OCCLUSION_V4_1
unzip -o ~/Downloads/BARAM_YOLO26X_MOT17_MOT20_AB.zip -d .
```

## 2. 공식 TrackEval 설치 (처음 한 번)

```bash
git clone --depth 1 https://github.com/JonathonLuiten/TrackEval.git ~/Downloads/TrackEval
```

공식 소스: https://github.com/JonathonLuiten/TrackEval

NumPy·SciPy가 설치된 기존 추론용 가상환경에서 실행합니다. 오류에 `ModuleNotFoundError`가 나타나면 다음을 실행하세요.

```bash
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python -m pip install numpy scipy
```

TrackEval 코드가 최신 NumPy와 충돌할 경우를 대비한 `baram_trackeval_compat.py`가 포함되어 있습니다. TrackEval 자체는 변경하지 않습니다.

## 3. 데이터셋 준비

**MOT17 train**과 **MOT20 train** 이미지 및 `gt/gt.txt` 파일이 필요합니다. 예를 들면:

```text
~/Downloads/MOT17/train/MOT17-09-FRCNN/
  ├── img1/000001.jpg ...
  ├── gt/gt.txt
  └── seqinfo.ini
~/Downloads/MOT20/train/MOT20-01/
  ├── img1/000001.jpg ...
  ├── gt/gt.txt
  └── seqinfo.ini
```

현재 스크립트는 `~/Downloads` 아래에서 `seqinfo.ini`, `img1/`, `gt/gt.txt`가 들어 있는 시퀀스를 자동 검색합니다. 다른 위치에 있으면 `--dataset-root "/데이터셋/상위폴더"`를 입력하세요.

MOT17은 동일 영상이 FRCNN/DPM/SDP 폴더로 반복되는 경우가 있습니다. 스크립트는 **한 장면당 한 폴더만 선택**하여 중복 집계를 방지하며, 가능하면 FRCNN을 사용합니다. MOT20의 각 고유 시퀀스는 각각 한 번 처리합니다.

## 4. 먼저 100프레임 테스트 (점수 평가하지 않음)

```bash
cd ~/Downloads/BARAM_LIVE_YOLO26X_OCCLUSION_V4_1
python3 run_baram_ab.py \
  --datasets MOT17 \
  --only MOT17-09-FRCNN \
  --quick-frames 100 \
  --skip-eval
```

위 명령은 실행 및 추적 결과 저장을 확인하기 위한 **스모크 테스트**입니다. HOTA/IDF1/MOTA/IDSW는 시퀀스 전체 추적 결과가 필요하므로 단축 실행으로는 산출하지 않습니다.

## 5. 본실험: MOT17 + MOT20, 두 모델

```bash
cd ~/Downloads/BARAM_LIVE_YOLO26X_OCCLUSION_V4_1
python3 run_baram_ab.py
```

YOLO26x와 OSNet 가중치를 두 모델이 동일하게 사용합니다. 두 모델 모두 `imgsz=640`, 일반 추적 검출 기준 `conf=0.40`, detector 저신뢰도 후보 하한 `0.15`로 YOLO를 실행합니다. Low-confidence rescue는 **V4.3에서만** 사용하고 baseline은 일반 검출 결과로만 추적합니다.

본실험은 **전체 시퀀스**를 모두 실행하므로 오래 걸릴 수 있습니다. 종료 후 아래 파일이 저장됩니다.

```text
~/Downloads/BARAM_SR_StrongSORT/outputs/BARAM_YOLO26X_STRONGSORT_AB/날짜시간/
  ├── comparison.csv       # MOT17/MOT20 모델별 종합 지표
  ├── comparison.md        # 발표 표(복사 가능)
  ├── per_sequence.csv     # 시퀀스별 성능 지표
  ├── all_runs.csv         # 각 시퀀스 FPS·Re-ID·VRAM 원자료
  ├── manifest.json        # 실행 조건과 코드 해시
  ├── trackers/            # 모델별 MOT 형식 예측 txt
  ├── gt/                  # GT 원본에 대한 심볼릭 링크(복사 없음)
  ├── metric_reports/      # 공식 TrackEval 로그와 점수
  └── runs/                # 프레임별 처리 시간, 요약, ID 이벤트
```

## 6. 경량형 V4.3까지 3가지 비교하기 (추천)

```bash
python3 run_baram_ab.py --include-selective
```

`v43_selective`는 불필요한 Re-ID 특징 추출 수를 줄이도록 설계되어 있습니다. 반면 `v43_full`은 `baseline`처럼 일반 검출마다 OSNet 특징을 추출하므로 Re-ID 횟수의 의미 있는 감소를 보장하지 않습니다.

## 7. 영상까지 저장 (선택)

```bash
python3 run_baram_ab.py --include-selective --video
```

각 트래커의 `annotated.mp4`를 생성합니다. 이 옵션을 쓰면 **FPS에 영상 인코딩 비용이 포함**됩니다. 발표용 벤치마크 표를 만들 때는 `--video` 없이 측정한 FPS를 사용하세요.

## 8. 지표 정의 및 공정성 조건

- **HOTA ↑**: 검출 품질·연관 정확도를 종합 평가 (TrackEval 공식 HOTA)
- **IDF1 ↑**: ID 연관 성능 (TrackEval Identity)
- **MOTA ↑**: 검출 오류와 ID 오류 등을 포함한 추적 성능 (TrackEval CLEAR)
- **IDSW ↓ / IDS ↓**: GT 기준 ID 전환 횟수 (TrackEval CLEAR `IDSW`)
- **FPS ↑**: 전체 프레임 수 ÷ 해당 프레임의 전체 처리 시간 합. 동일 머신, GPU, 시퀀스, 모델 해상도에서 비교. 이미지 로딩·YOLO·OSNet·데이터 연결·ID 로직·MOT 파일 저장은 포함하고 **화면 렌더링과 MP4 인코딩은 제외**합니다. 초기 가중치 로드/워밍업은 제외합니다.
- **Re-ID feature extractions ↓**: OSNet에 투입한 검출 크롭의 실제 총개수 (추론 호출 회수와는 다름). 같은 검출 모수 대비 비교하려면 `ReID_per_detection_pct`도 참고하세요.
- **Peak VRAM**: `nvidia-smi`가 추적 프로세스를 약 0.5초 간격으로 샘플링한 값. 별도 프로세스가 없는 상태에서 비교하세요.

V4.3에서 신원이 불확실해 `VERIFY`가 붙은 프레임은 MOT 결과에서 제외됩니다. 따라서 IDSW가 줄더라도 MOTA/IDF1/HOTA가 떨어질 수 있습니다. **IDSW만 가지고 모델이 더 좋다고 판단하면 안 됩니다.**

TrackEval 점수는 공개 정답이 있는 **train 분할을 대상으로 해당 시퀀스를 자체 평가한 결과**입니다. 공식 MOTChallenge 숨겨진 test 리더보드와 동일한 값이 아니므로 표 제목에 반드시 `MOT17-train`, `MOT20-train`을 표시하세요.

## 9. 문제가 발생할 때

- `No GT-labeled ...`: `--dataset-root` 경로를 지정하거나 MOT20 train 세트를 내려받으세요.
- `TrackEval not found`: 2번 설치를 수행하세요.
- 모델 추적이 실패: `runs/<dataset>/<sequence>/<mode>/terminal.log` 확인.
- 점수 평가만 실패: `metric_reports/<dataset>/<mode>/trackeval.log` 확인. 추적 `mot_results.txt`와 FPS/Re-ID는 남습니다.
- GPU 실행 오류: 아래 가상환경 테스트부터 확인하세요.

```bash
~/Downloads/BARAM_SR_StrongSORT/.venv/bin/python -c 'import torch, ultralytics; print(torch.cuda.is_available(), ultralytics.__version__)'
```

**유의:** 코드 문법 및 실행 오케스트레이터의 일부 단위 테스트만 로컬에서 확인했습니다. 이 환경에는 사용자의 NVIDIA GPU 및 MOT17/20 실데이터가 없어 HOTA/IDF1/MOTA/IDS/FPS 실측치는 아직 없습니다. 제공 스크립트를 Ubuntu에서 실행해야 결과가 나옵니다.
