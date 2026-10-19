# AInAlgorithm
멀티 로봇 WAAM 작업 배정 및 Makespan 최소화 프로젝트

## Final Algorithm

최종 코드와 공개 모델 6개의 trajectory.csv는 [final_algorithm](final_algorithm/)에 있습니다.

- [코드와 실행 안내](final_algorithm/README.md)
- [Model01](final_algorithm/model01/trajectory.csv) · [Model02](final_algorithm/model02/trajectory.csv) · [Model03](final_algorithm/model03/trajectory.csv)
- [Model04](final_algorithm/model04/trajectory.csv) · [Model05](final_algorithm/model05/trajectory.csv) · [Model06](final_algorithm/model06/trajectory.csv)

최종 방법은 PPO를 호출하지 않는 휴리스틱입니다. 기존 sample 폴더는 과거 PPO 자료이며, 보고서는 별도로 제출합니다.

## 보고서 비교 기준

[비교용 CSV와 공식 판정 기록](comparison_baselines/README.md): 개선 전 6개 모델 기록 및 PPO 대 Greedy 비교. 최종 제출 CSV와 분리하여 보관합니다.

## 2026 10 19 최종 교체

고정 경계 LNS를 반영했습니다. 6개 전체 공식 PASS이며 10/15 고정본보다 5개 모델이 개선됐습니다. Model05는 100633.805410초에서 96424.118129초로 4209.687280초(4.183%) 줄었습니다. Model06은 기존 결과를 유지했습니다. [상세 결과와 실행 방법](final_algorithm/README.md). 보고서는 저장소에 업로드하지 않습니다.
