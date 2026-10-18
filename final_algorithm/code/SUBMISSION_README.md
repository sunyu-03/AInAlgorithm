# Final Algorithm 실행 안내

최종 방법은 예약 기반 스케줄링과 국소 탐색을 결합한 휴리스틱입니다. PPO 정책이나 체크포인트를 사용하지 않습니다. 기존 PPO 비교 실험은 저장소의 sample 폴더에 보존돼 있습니다.

사용자 요청에 따라 이 폴더에는 코드와 공개 Model01~06의 최종 trajectory.csv만 제공합니다. 보고서, 보고서 첨부, 실험 로그, 검증 결과, 과거 비교 입력은 로컬에 보관합니다. 모델별 CSV는 해당 모델의 제공된 원본 target.stl 및 config.yaml과 함께 공식 Validator에 입력합니다. STL/config를 다른 모델의 파일로 바꾸거나 수정하지 않습니다.

## 폴더

- `code/`: 2026-10-15 최종 고정 소스. 단계별 탐색 실행 파일과 포함된 Validator 코드, requirements.txt.
- `model01/` ~ `model06/`: 해당 공개 모델의 trajectory.csv.

## 실행 조건

검증에 사용한 환경은 Python 3.13.3, WAAM Validator 1.0.1입니다. 의존성 버전은 requirements.txt에 있습니다. 아래 예시는 현재 검증된 Windows PC의 Python과 기존 의존성을 사용합니다. 새 PC의 전체 설치 검증은 아직 수행하지 않았습니다.

```powershell
$env:WAAM_DEPS = 'C:\Users\User\Desktop\AInAlgorithm\waam_generalization\integration\.deps'
$waamPython = 'C:\Users\User\Desktop\WAAM_Validator-main\.venv\Scripts\python.exe'
```

## 새 입력에 단계별 적용

저장소 루트에서 실행하며, `my_job`에 사용할 target.stl과 config.yaml을 준비합니다. 출력 폴더는 매번 존재하지 않는 새 경로여야 합니다. 각 단계가 PASS일 때 다음 단계로 진행합니다.

```powershell
& $waamPython -B .\final_algorithm\code\optimize.py --job .\my_job --archive .\runs\empty_archive --out .\runs\initial
& $waamPython -B .\final_algorithm\code\run_two_layer_lookahead.py --baseline .\runs\initial\selected --out .\runs\lookahead
& $waamPython -B .\final_algorithm\code\run_bundle_stay.py --baseline .\runs\lookahead\selected --out .\runs\bundle24
& $waamPython -B .\final_algorithm\code\run_last_bundle36.py --baseline .\runs\bundle24\selected --search-input .\runs\lookahead\selected --control-run .\runs\bundle24 --out .\runs\bundle36
```

마지막 두 실행은 같은 검색 입력에서 2·4 및 3·6 묶음을 비교합니다. 더 느리거나 실패한 후보는 채택하지 않고 기존 PASS를 보존합니다. 이는 새로운 입력에 고정 구성요소를 연결하는 예시이며, 공개 6개에서 누적한 모든 과거 개선 단계와 초기 경로를 다시 재현한다는 뜻은 아닙니다. 공개 모델 제출값은 위 model01~model06의 보존 CSV입니다.

소스에는 과거 실험과 마지막 실험 재현용 실행 파일도 포함돼 있습니다. `verify_submission.py`와 고정본 재현 명령은 로컬에 보존한 FROZEN_VERSION.json, 비교 입력 및 검증 기록을 요구합니다. 사용자의 최소 업로드 범위에 따라 그 자료는 이 저장소에 추가하지 않았습니다.

## 결과 해석

공개 6개 최종 CSV는 보존 공식 검증에서 모두 PASS입니다. 전역 최적성과 비공개 일반화는 증명하지 않았습니다. 최종 휴리스틱이 학습 모델이 아니므로 추가 checkpoint가 필요하지 않습니다. `.gitattributes`는 Git의 줄바꿈 변환으로 원본 코드와 CSV 바이트가 바뀌지 않도록 설정합니다.

## 비교 자료 추가

2026-10-19 요청에 따라 보고서의 비교용 CSV와 보존 공식 판정 기록을 저장소 루트의 [comparison_baselines](../../comparison_baselines/README.md)에 별도로 공개했습니다. 위 최소 업로드 안내는 최초 업로드 시점의 범위이며, 최종 코드와 제출 CSV는 변경하지 않았습니다. 보고서는 업로드하지 않습니다.
