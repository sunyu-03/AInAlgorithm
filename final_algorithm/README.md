# Final Algorithm

예약 기반 휴리스틱으로 초기 시간표를 만들고, 고정 경계 LNS 방식으로 병목 구간의 배정·순서·방향을 개선합니다. 최종 단계는 PPO 정책이나 checkpoint를 호출하지 않습니다. 이전 코드와 CSV는 Git 이력에 보존돼 있습니다.

## 제출 결과

모든 최종 CSV는 제공된 원본 STL/config 조건에서 공식 Validator 1.0.1 전체 PASS입니다. 모델 번호에 따른 탐색 분기는 없습니다. 아래는 10/15 고정본과의 비교이며 전역 최적값을 증명한 결과가 아닙니다.

| 모델 | 10/15 고정본 초 | 최종 초 | 단축 초 | 단축률 | 판정 |
|---|---:|---:|---:|---:|---|
| 01 | 696.544503 | 694.254345 | 2.290158 | 0.329% | PASS |
| 02 | 1629.199019 | 1621.914318 | 7.284700 | 0.447% | PASS |
| 03 | 1174.607215 | 1173.872394 | 0.734821 | 0.063% | PASS |
| 04 | 26127.124100 | 26085.982206 | 41.141894 | 0.157% | PASS |
| 05 | 100633.805410 | 96424.118129 | 4209.687280 | 4.183% | PASS |
| 06 | 3996.892940 | 3996.892940 | 0.000000 | 0.000% | PASS |

Model 01/02의 실제 LNS 시작값은 직전 개선이 반영된 694.982918/1628.519634초입니다. 따라서 위 총 단축에는 LNS 이전 개선 1.561585/0.679385초도 포함됩니다. 적층 선의 위치·길이·개수와 STL/config를 유지했고 진행 방향은 바꿀 수 있습니다. 형상 IoU 변화는 계산 오차 수준입니다. Model 06은 개선을 찾지 못해 기존 CSV를 보존했습니다.

## 구조

- optimize_schedule.py: 공통 다섯 단계 후처리 진입점
- code/: 방향 선택을 허용하는 LNS와 기존 초기 경로 생성·예약 스케줄러
- code_fixed/: 방향을 고정하는 첫 두 단계 코드
- model01~model06/trajectory.csv: 최종 제출 CSV
- ../comparison_baselines/frozen_20261015/: 이번 추가 개선의 비교 기준 CSV
- ../comparison_baselines/pre_lns_20261019/: 실제 후처리 시작 CSV

STL/config, 실험 로그와 checkpoint는 이번 교체에 추가하지 않았습니다. 최종 보고서는 [reports](../reports/INE4107_midterm_report_final.docx)에 추가했습니다. 모델별 CSV를 해당 공개 모델의 원본 target.stl/config.yaml과 함께 공식 Validator에 입력합니다.

## 현재 검증된 PC에서 실행

기존 WAAM Python 환경과 의존성 폴더를 사용합니다. 새 PC 전체 설치 검증은 수행하지 않았습니다. 공통 후처리 입력은 target.stl/config.yaml 및 공식 PASS trajectory.csv가 들어 있는 폴더입니다. 출력은 새 폴더여야 합니다. 저장소 루트의 VS Code 터미널 예시입니다.

```powershell
$env:WAAM_DEPS = 'C:/Users/User/Desktop/AInAlgorithm/waam_generalization/integration/.deps'
$waamPython = 'C:/Users/User/Desktop/WAAM_Validator-main/.venv/Scripts/python.exe'
& $waamPython -B ./final_algorithm/optimize_schedule.py --input ./my_job --out ./runs/lns_new
```

처리 순서는 방향 고정 탐색 → 방향 고정 규칙 확장 → 방향 선택 탐색 → 방향 선택 규칙 확장 → 동일 실행에서 검증된 과거 성공 규칙 재사용 1회입니다. 각 단계는 전체 공식 PASS이며 1μs를 초과해 빠른 후보만 채택합니다. 실패하거나 느린 후보는 이전 PASS로 되돌립니다. 성공 규칙이 없으면 해당 확장을 건너뜁니다. 성공 규칙은 이번 입력의 탐색 기록이며 학습된 모델이 아닙니다.

초기 탐색은 모델당 120초, 병목 상위 최대 12개 층, 재배정 대상 10/20/40개, 공간 분위수 25/50/75%, 창별 경로 후보 6개, 로봇 우선순위 6개를 사용합니다. 규칙 확장은 최대 3개 규칙과 평가 1200회, 비상 시간 제한 1200초로 제한합니다. 시간 제한 탐색이므로 CPU 부하에 따라 방문 후보와 최종 결과가 달라질 수 있습니다. 저장된 최종 CSV가 이번 측정값의 제출 해입니다.

## STL/config에서 초기 시간표 만들기

후처리 전에 초기 PASS가 필요합니다. 새 STL/config는 기존 구성요소로 초기 경로를 생성한 후 후처리에 연결합니다. 다음은 구성요소 연결 예시이며, 공개 6개에 누적된 모든 과거 탐색을 동일하게 재현하는 명령은 아닙니다. 각 단계의 run.json에서 PASS를 확인합니다.

```powershell
& $waamPython -B ./final_algorithm/code/optimize.py --job ./my_shape --archive ./runs/empty_archive --out ./runs/initial
& $waamPython -B ./final_algorithm/code/run_two_layer_lookahead.py --baseline ./runs/initial/selected --out ./runs/lookahead
& $waamPython -B ./final_algorithm/code/run_bundle_stay.py --baseline ./runs/lookahead/selected --out ./runs/bundle24
& $waamPython -B ./final_algorithm/code/run_last_bundle36.py --baseline ./runs/bundle24/selected --search-input ./runs/lookahead/selected --control-run ./runs/bundle24 --out ./runs/bundle36
& $waamPython -B ./final_algorithm/optimize_schedule.py --input ./runs/bundle36/selected --out ./runs/lns_new
```

국소 구간만 안전해도 전체 판정이 자동 PASS라고 가정하지 않습니다. TCP·팔 점유 예약과 경계 보존은 후보 생성 조건이며, 최종 채택에는 전체 공식 Validator를 사용합니다. 실제 관절각·실물 제조나 미사용 비공개 형상의 성능을 입증한 결과는 아닙니다.
