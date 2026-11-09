# Selective Re-ID StrongSORT — 프로젝트 목적 및 실험 결과

> 2026년 BARAM 후반기 프로젝트. YOLO26x 기반 후속 실험 요약. 아래 서로 다른 측정 경로의 FPS는 혼합하지 않는다.

## 1. 연구 배경과 목적

기본 StrongSORT는 검출된 다수의 객체에 대해 OSNet 외형 특징을 지속적으로 계산하여 객체 간 동일성을 판단한다. 사람 수가 많아질수록 외형 특징 추출 비용이 증가하고 실시간 추적에 부담이 생길 수 있다.

이 프로젝트의 목적은 **기존 StrongSORT의 모션 기반 추적 구조를 활용하면서, 구분이 모호한 객체에만 Re-ID 특징을 선택적으로 추출**하여 연산량과 GPU 메모리 사용량을 줄이고, 가능한 한 객체 ID의 연속성을 유지하는 것이다. 따라서 단순 속도 증가뿐 아니라 추적 정확도(HOTA/MOTA), ID 일관성(IDF1/IDSW), 추출 횟수와 VRAM을 함께 비교한다.

## 2. 접근 방법

1. **객체 탐지:** YOLO26x로 프레임별 사람 검출.
2. **모션 예측:** Kalman filter를 이용해 트랙 위치 예측.
3. **후보 제한:** Mahalanobis distance 기반 motion gating.
4. **모호성 분석:** 예측/검출 박스 간 IoU, 종횡비 유사도(ARS), 경쟁 후보 등을 이용해 외형 비교 필요 여부 판단.
5. **선택적 외형 특징 추출:** 모호한 연관 후보에만 OSNet Re-ID 사용. 명확한 후보는 모션·기하 정보로 처리.
6. **연관 및 ID 보완:** Hungarian matching, Appearance/Identity Memory, Live Relink, Occlusion Rescue, ID Verification 등을 실험.
7. **평가:** 동일 검출 조건에서 StrongSORT baseline과 Selective 변형 비교. TrackEval의 HOTA/CLEAR/Identity와 FPS, Re-ID 호출 수, VRAM 측정.

이후 Identity Memory, Live Relink, Occlusion Rescue의 기여도를 분리하기 위해 ablation 실험도 수행했다.

## 3. 주요 비교 결과 — 최종 포스터 집계

측정 대상: MOT17/MOT20 **train** 구간. **공식 test 리더보드 점수가 아님.** 아래 수치는 2026-11-07 프로젝트 포스터에 수록된 집계값이며, 파일은 `experiments/poster_comparison.csv`에도 기록했다.

| 데이터셋 | 방법 | HOTA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | FPS ↑ | Re-ID 추출 ↓ | VRAM MiB ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MOT17 | StrongSORT | 39.74 | 45.83 | 32.67 | 243 | 16.98 | 47,881 | 822 |
| MOT17 | SR-StrongSORT | 37.12 | 43.02 | 31.75 | 359 | 17.54 | 27,865 | 766 |
| MOT20 | StrongSORT | 10.38 | 7.67 | 4.33 | 232 | 17.24 | 59,283 | 1,034 |
| MOT20 | SR-StrongSORT | 12.95 | 11.17 | 7.99 | 381 | 18.03 | 29,478 | 872 |

### 주요 변화

| 지표 | MOT17 | MOT20 |
|---|---:|---:|
| Re-ID 추출 횟수 | 41.8% 감소 | 50.3% 감소 |
| GPU VRAM 사용량 | 6.8% 감소 | 15.7% 감소 |
| HOTA | -2.62p | +2.57p |
| IDF1 | -2.81p | +3.50p |
| IDSW | 243 → 359 (증가) | 232 → 381 (증가) |

**해석:** Selective Re-ID는 두 환경 모두에서 특징 추출량과 GPU 메모리 사용량을 줄였다. 밀집된 MOT20에서는 HOTA와 IDF1이 상승했지만, MOT17에서는 감소했다. 특히 두 데이터셋 모두 IDSW가 증가했으므로 **기존 StrongSORT보다 모든 추적 성능이 개선되었다고 결론 내릴 수 없다.**

## 4. 2026-11-06 원본 콘솔 로그와의 비교

11월 6일 YOLO26x 전체 비교 콘솔 로그에는 다음과 같이 별도의 FPS 측정값이 남아 있다. 포스터와 실행 경로 또는 설정 차이가 확인되지 않았으므로 각각 별도 기록한다.

| 데이터셋 | 방법 | FPS | Re-ID 추출 | Peak VRAM MiB |
|---|---|---:|---:|---:|
| MOT17 | baseline | 17.56 | 47,881 | 822 |
| MOT17 | v43_full | 16.59 | 47,881 | 822 |
| MOT17 | v43_selective | 18.73 | 27,865 | 766 |
| MOT20 | baseline | 16.81 | 59,283 | 1,034 |
| MOT20 | v43_full | 14.54 | 59,283 | 1,034 |
| MOT20 | v43_selective | 16.31 | 29,478 | 872 |

**주의:** 이 콘솔 로그의 TrackEval 실행은 `SEQMAP_FILE`에 list가 전달되어 `TypeError: stat: path should be string ... not list`로 실패했다. 따라서 **이 실행에서 HOTA/IDF1/MOTA/IDSW가 정상 계산되었다고 표시해서는 안 된다.** 포스터에 기재된 지표는 별도 집계로 보존하며 해당 TrackEval 산출 파일을 확보할 때까지 재검증 대상으로 취급한다.

이 원본 로그의 FPS만으로 비교하면, Selective와 baseline의 차이는 MOT17 약 **+6.7%**, MOT20 약 **-3.0%**다. 포스터의 FPS는 MOT17 **+3.3%**, MOT20 **+4.6%**이므로 서로 다른 측정을 섞어 하나의 결론으로 제시하지 않는다.

## 5. V4.4 및 V4.5 진단

- **V4.4 IDSW Guard:** ID 전환을 억제하는 방향으로 조정했다. 이전 결과 정리에는 V4.3 대비 IDSW 감소와 HOTA/IDF1 하락이 함께 기록되어 있다. 원본 ZIP/측정 파일 미확보로 재검증이 필요하다.
- **V4.5 Diagnostic Ablation:** Identity Memory, Live Relink, Occlusion Rescue 제거 효과를 개별 진단했다. 이전에 공유된 진단 표 일부는 `experiments/v45_diagnostic_excerpt.csv`에 구분해 보존했다. 이 표는 포스터의 전체 비교와 실험 구성이 달라 직접 합산하지 않는다.
- MOT17 진단 발췌에서 Identity Memory를 끄면 HOTA **37.75 → 33.95**, IDF1 **47.48 → 39.80**, IDSW **110 → 139**였다.
- MOT17 진단 발췌에서 Occlusion Rescue를 끄면 HOTA **37.75 → 41.13**, IDF1 **47.48 → 54.67**, IDSW **110 → 71**로 나타났다.
- MOT20 진단 발췌에서는 Occlusion Rescue를 끄면 HOTA **25.01 → 22.27**, IDF1 **31.59 → 26.55**, IDSW **186 → 168**로 나타났다.
- `no_live_relink`의 MOT17 주요 정확도 값이 v43과 같아 해당 실험에서 모듈 활성화 여부와 사례 수를 다시 확인할 필요가 있다.

V4.5 진단 표는 이전에 공유된 실행 결과를 발췌한 것이며 원본 ZIP·전체 CSV 및 환경정보까지 다시 확인한 상태는 아니다. **단일 실험 반복 횟수와 표본 조건을 고려하지 않고 통계적 우월성을 단정할 수 없다.**

## 6. 결론 및 후속 과제

1. **연산량 경량화는 확인:** 높은 Re-ID 추출 감소율과 VRAM 절감.
2. **추적 정확도는 혼재:** MOT20 HOTA/IDF1 개선, MOT17 하락.
3. **ID 안정성은 핵심 과제:** 전반적인 IDSW 증가와 버전별 성능 상충 확인.
4. **향후 과제:** 모호성 판단 임계값 튜닝, 가림/재등장별 선택적 특징 추출, 잘못된 ID 복원 방지, 동일 조건 반복 측정, 공식 TrackEval 재검증.

## 7. 재현성 및 소스 범위

- 이 저장소의 `baram/` 구현은 **2026-10-28에 보존된 코드 스냅샷**이다. YOLO26x 기반 V4.3·V4.4·V4.5 실험 패키지 코드를 원본 그대로 통합한 상태가 아니다.
- 해당 패키지들은 이전 프로젝트 파일 목록에서 확인되지만, 현재 원본 ZIP 바이트를 읽을 수 없어 정확한 소스 병합·diff·실행 검증을 진행하지 못했다. 상태는 `CODE_STATUS.md`에 기록한다.
- 원본 데이터셋, pretrained checkpoints, 실험별 MOT 예측과 전체 TrackEval 산출 로그는 저장소에 포함되어 있지 않다.
- **데이터 출처 구분:** `poster_comparison.csv`(포스터의 집계), `run_20261106_comparison.csv`(실제 콘솔 로그 수치, TrackEval 실패), `v45_diagnostic_excerpt.csv`(이전 공유된 진단 결과의 일부, 재확인 필요).
