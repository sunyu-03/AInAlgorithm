# 물체 01 알고리즘 — code_ver.1

공개 모델 01(Honeycomb)의 중심선 30개를 작업으로 정의하고, PPO가 각 작업을
로봇 1·2·3 중 하나에 배정한다. 이는 1주차 `AssignmentEnv`의
`행동 = 담당 로봇 선택` 개념을 그대로 반영한 `Discrete(3)` 환경이다.

## 현재 스케줄 방식

- 6개 레이어를 아래에서 위로 처리한다.
- PPO 관측은 현재 작업 위치·길이, 진행률, 로봇별 작업량·작업 수·거리로 구성한다.
- PPO 행동 `0, 1, 2`는 각각 로봇 `1, 2, 3` 선택이다.
- 보상은 makespan 감소가 중심이며 거리와 작업량 불균형을 작은 보조항으로 사용한다.
- 같은 로봇에 배정된 선분은 가까운 순서로 정렬한다.
- `arc_on_time_s`, `arc_off_time_s`를 W 구간으로 포함한다.
- 첫 버전은 충돌 없는 결과를 우선하기 위해 한 번에 한 로봇만 가동한다.
- 제공된 `config.yaml`, `target.stl`은 수정하지 않고 그대로 복사한다.

## 실행

```powershell
cd "C:\Users\User\Desktop\AInAlgorithm\sample\0929\code_ver.1"
& "C:\Users\User\Desktop\WAAM\TEAM_PROJECT\week1\.venv\Scripts\python.exe" -B train_ppo_model_01.py
```

학습 후 공식 Validator 결과 파일을 만들려면 다음을 실행한다.

```powershell
& "C:\Users\User\Desktop\WAAM_Validator-main\.venv\Scripts\waam-validator.exe" run `
  "..\1\my_job" --output "..\1\validation_run" --json
```

생성 입력은 `..\1\my_job`, 검증 결과는 `..\1\validation_run`에 저장된다.
검증 폴더가 이미 비어 있지 않으면 과거 증거를 덮어쓰지 않고 중단한다.

## 다음 버전에서 개선할 점

이 버전은 안전한 PASS 기준선이므로 세 로봇을 직렬로 사용한다. 다음 버전에서는
동시에 움직여도 충돌하지 않는 작업 묶음을 찾아 makespan을 줄인다.

`run_model_01.py`의 greedy는 PPO 성능을 비교하고 Validator 연결을 확인하기 위한
기준선으로만 사용한다. 최종 `trajectory.csv`는 학습된 PPO 정책으로 다시 생성한다.

## code_ver.1 실제 PPO 결과

- 학습: PPO, `30,000` timestep, seed `42`
- 입력 검사: PASS (`3 robots`, `640 rows`)
- 전체 Validation: PASS
- Greedy 기준 Makespan: `1654.836505404 s`
- PPO Makespan: `1643.830117223 s`
- PPO 개선: `11.006388181 s`
- Coverage: `0.9999996053`
- Overfill ratio: `0.00000000046`
- IoU: `0.9999994260`
- 실패 레이어: `0`
- 충돌 이벤트: `0`
- XY Reach: PASS

학습 모델은 `ppo_model_01.zip`, 학습 조건과 로봇 배정 결과는
`ppo_training_summary.json`에 저장된다.
