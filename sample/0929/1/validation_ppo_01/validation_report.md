# WAAM Validator Report

## Overall Result

**PASS**

- WAAM Validator version: 1.0.1

## Input Summary

- Directory: `C:\Users\User\Desktop\AInAlgorithm\sample\0929\1\my_job`
- Trajectory rows: 640
- Target watertight: True

## Schedule

- Makespan: 1643.83 s
- R1: completion 1643.83 s; Deposition 315.00 s; Travel 111.16 s; Wait 1217.67 s
- R2: completion 1643.83 s; Deposition 189.00 s; Travel 112.47 s; Wait 1342.36 s
- R3: completion 1643.83 s; Deposition 441.00 s; Travel 115.20 s; Wait 1087.63 s

## Robot XY Reach

- Passed: True
- R1: max XY distance 1402.52 / 1500.00 mm; XY margin 97.48 mm; violating points 0
- R2: max XY distance 1363.79 / 1500.00 mm; XY margin 136.21 mm; violating points 0
- R3: max XY distance 1402.52 / 1500.00 mm; XY margin 97.48 mm; violating points 0

## Collision

- Arm Envelope events: 0
- Minimum Arm Envelope safety margin: 678.20 mm
- Arm centerline distance at worst case:
  928.20 mm
- Arm required centerline distance at worst case:
  250.00 mm
- TCP_RADIUS events: 0
- Minimum TCP distance: 928.20 mm

## Shape

- Target layer-integrated volume: 116840.445 mm³
- Deposited volume: 116840.420 mm³
- Intersection volume: 116840.399 mm³
- Underfill volume: 0.046 mm³
- Overfill volume: 0.000 mm³
- Coverage: 100.00%
- Underfill: 0.00%
- Overfill: 0.00%
- IoU: 100.00%
- Failed layers: 0 / 6

## Failure Reasons

None

## Warnings

None

## Validation Violations

These findings contribute to a normal `FAIL`; they do not mean that the pipeline ended with
the fatal status `ERROR`.

None

## Output Files

- `summary.json`
- `validation_report.md`
- `robot_metrics.csv`
- `collision_events.csv`
- `layer_metrics.csv`
- `run.log`
- `validation_inputs.json`

## Interpretation Limitations

본 검증기는 각 로봇을 폭이 고정된 Base–TCP 2D Capsule로 단순화하고, XY 평면상 Capsule 안전 여유와 TCP 허용 원 침범만을 로봇 간 충돌로 판단한다. 검사는 adaptive sample 기반이며 연속시간 swept collision을 보증하지 않는다. 실제 로봇 링크, 관절 자세, Z 방향 분리, 지그 및 환경 충돌은 반영하지 않는다. 형상 검증은 일정한 비드 폭과 layer 높이를 가정한 명목 기하 모델이며 열변형, 비드 형상 변화, 용융풀 거동 및 공정 불안정성을 예측하지 않는다.
