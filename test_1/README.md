# WAAM 멀티 로봇 개인 실험 test_1

`TEAM_PROJECT/week1`은 제출 당시 상태로 보존하고, 이후 개인 실험은 WAAM 루트에
`test_1`, `test_2`, ...처럼 새 폴더를 만들어 진행한다.

이 첫 버전은 교수님 제공 모델 6개의 입력 구조와 형상 복잡도를 공식 Validator로
확인한다.

## 이번 버전의 범위

1. `01`~`06` 모델에 `config.yaml`과 `target.stl`이 모두 있는지 검사한다.
2. 공식 Validator와 동일한 로더로 STL을 읽는다.
3. 크기, 삼각형 수, 적층 층 수, 층별 단면의 연결 영역과 구멍 수를 기록한다.
4. 결과를 `results/model_inventory.json`에 저장한다.

아직 `trajectory.csv`를 생성하지 않는다. 경로 생성 알고리즘은 이 조사 결과를 바탕으로
다음 단계에서 추가한다.

## 실행

```powershell
cd "$env:USERPROFILE\Desktop\WAAM\test_1"
& "..\TEAM_PROJECT\week1\.venv\Scripts\python.exe" -B inspect_models.py
```

기본 입력 위치는 `C:\Users\User\Desktop\WAAM_Validator-main\tests`이다. 다른 위치를
사용하려면 `--models` 옵션을 준다.

```powershell
& "..\TEAM_PROJECT\week1\.venv\Scripts\python.exe" -B inspect_models.py --models "모델_폴더"
```

