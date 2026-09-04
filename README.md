# SO-101 pick RL

SO-101이 큐브를 집어 들어 올리는 정책을 Isaac Lab의 병렬 PPO로 학습하고, 같은 성공 조건을 MuJoCo에서 다시 평가하는 프로젝트입니다. 학습 속도만 비교하지 않고 보상 설계, 접촉 물리, 환경 수, 시뮬레이터 차이가 결과에 어떤 영향을 줬는지 실행 기록으로 남깁니다.

현재 상태는 프로젝트 계약과 실행 문서를 만든 단계입니다. SO-101 PPO 학습과 MuJoCo 교차평가는 아직 실행하지 않았습니다.

## 풀려는 문제

로봇팔 강화학습 예제는 학습 영상을 만드는 데서 끝나기 쉽습니다. 이 프로젝트는 아래 질문을 실험으로 답합니다.

1. SO-101의 6개 관절을 position-delta action으로 제어해 큐브를 접촉만으로 들어 올릴 수 있는가?
2. 환경 수를 늘릴 때 RTX 3060 12GB에서 처리량이 어디까지 좋아지는가?
3. 접근, 파지, 들어 올리기 보상 가운데 정책 행동을 실제로 바꾸는 항목은 무엇인가?
4. Isaac에서 학습한 정책이 MuJoCo에서도 같은 성공 판정을 통과하는가?
5. 물리 파라미터 랜덤화가 held-out 조건과 다른 시뮬레이터에서의 성능에 어떤 영향을 주는가?

## 범위

첫 번째 버전은 상태 기반 PPO만 다룹니다. 시뮬레이터가 알고 있는 관절 상태와 물체 자세를 정책 입력으로 사용하므로 결과는 simulation-only로 표시합니다. 카메라 기반 IL과 실물 SO-101 실행은 기존 `so101-mobile-manipulation` 프로젝트에서 별도로 다룹니다.

```text
Windows 11, RTX 3060 12GB
SO-101 asset -> Isaac Lab task -> RSL-RL PPO -> checkpoint
                                              |
                                              v
Ubuntu
SO-101 MJCF -> MuJoCo evaluator --------> cross-sim report
```

## 왜 별도 저장소인가

`isaac-walk-rl`은 Go2 보행 PPO의 보상과 지형 실험을 설명합니다. `so101-mobile-manipulation`은 실물 팔, 손목 카메라, LeRobot 데이터와 ACT를 다룹니다. 이 저장소는 물체 접촉이 포함된 팔 강화학습과 Isaac-MuJoCo 차이를 한 실험으로 묶습니다. 세 프로젝트의 질문과 완료 근거를 섞지 않습니다.

## 저장소 지도

| 경로 | 내용 |
| --- | --- |
| `common/task_spec.json` | 관절 순서, 단위, 성공 판정, 평가 횟수의 정본 |
| `isaaclab/` | Windows용 SO-101 Isaac Lab extension과 RSL-RL 설정 |
| `mujoco/` | Ubuntu용 SO-101 MuJoCo 환경 |
| `evaluation/` | 공통 평가와 비교 도구 |
| `reports/` | 작은 실행 manifest, 결과 표, 그래프 |
| `HANDOFF.md` | 다른 PC 또는 새 작업 세션에 전달할 첫 문서 |
| `docs/TRANSFER.md` | Windows로 저장소를 옮기는 방법 |
| `docs/HANDOFF_WINDOWS.md` | Windows 실행자에게 넘길 작업 지시서 |
| `docs/HANDOFF_LINUX.md` | Ubuntu 실행자에게 넘길 작업 지시서 |
| `docs/EXPERIMENT_PROTOCOL.md` | 대조군, seed, 지표, 중단 조건 |
| `docs/PORTFOLIO_EVIDENCE.md` | 포트폴리오에 쓸 수 있는 주장과 필요한 증거 |

## 시작

두 PC에서 clone한 직후 공통 계약부터 확인합니다.

```bash
python scripts/validate_contract.py
python -m unittest discover -s tests -v
```

다른 PC로 옮길 때는 [저장소 전달 방법](docs/TRANSFER.md)을 먼저 봅니다. 그다음 [통합 인계서](HANDOFF.md)에서 맡은 PC의 경로로 들어갑니다. Windows는 [Windows 인계 문서](docs/HANDOFF_WINDOWS.md), Ubuntu는 [Ubuntu 인계 문서](docs/HANDOFF_LINUX.md)를 따릅니다. 전체 순서와 관문은 [프로젝트 계획](docs/PROJECT_PLAN.md)에 있습니다.

## 포트폴리오 완료 조건

다음 자료가 모두 있을 때 프로젝트를 완료로 표시합니다.

- 1, 64, 256, 512 환경의 reset/step 및 처리량 기록
- 같은 예산으로 실행한 PPO 3개 seed
- reward ablation과 Domain Randomization 대조표
- ID 조건과 held-out 조건의 성공률
- Isaac checkpoint를 MuJoCo에서 평가한 결과
- 실패 사례 영상과 원인 분류
- 코드 commit, 설정, checkpoint, 결과 파일을 잇는 manifest

성능이 기대에 못 미쳐도 원인을 물리, 관측, 보상, 학습 안정성으로 나눠 재현할 수 있으면 결과로 남깁니다.

## 참고 구현

- [Isaac Lab reinforcement learning](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Isaac Lab 외부 프로젝트 템플릿](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/developer-guide/template.html)
- [ROBOTIS cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)
- [LeIsaac SO-101 환경](https://github.com/LightwheelAI/leisaac)
- [LeIsaac 설치와 버전 호환표](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)
