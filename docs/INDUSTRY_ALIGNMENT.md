# 강화학습 직무와의 연결

## 이 프로젝트가 겨냥하는 직무

이 저장소는 로봇 시뮬레이션과 강화학습 엔지니어 역할을 겨냥합니다. 실물 제품의 전체 제어 스택이나 카메라 기반 로봇 파운데이션 모델을 한 저장소에서 모두 보여주려 하지 않습니다.

| 프로젝트 작업 | 직무에서 확인할 수 있는 능력 | 남길 증거 |
| --- | --- | --- |
| SO-101 외부 task 작성 | 로봇 자산과 학습 환경 통합 | asset provenance, task registry, reset test |
| 관측과 action 계약 | 제어 인터페이스 설계 | schema, fixture, 단위 테스트 |
| GPU 병렬환경 사다리 | 자원 profiling과 규모 선택 | VRAM, steps/s, wall time |
| RSL-RL PPO | 정책 학습과 수렴 진단 | config, curves, 3-seed 결과 |
| reward ablation | 원인 분리 실험 | one-factor 비교표 |
| Domain Randomization | 분포 설계와 held-out 평가 | DR manifest, ID/OOD 결과 |
| Isaac-MuJoCo 교차평가 | simulator 종속성 진단 | 같은 정책의 cross-sim 결과 |
| run manifest | 재현성과 artifact 관리 | commit, config, checkpoint SHA |

Isaac Lab은 GPU 병렬 locomotion과 manipulation에 RSL-RL을 기본 선택으로 안내합니다. NVIDIA의 Franka Lift 예제는 상태 기반 PPO를 scratch에서 학습합니다. ROBOTIS `cyclo_lab`도 OMY Lift를 512환경 PPO로 제공하고 IL은 별도 경로로 둡니다.

참고 자료:

- [Isaac Lab RL workflows](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Isaac Lab project template](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/developer-guide/template.html)
- [Isaac Lab Arena Franka Lift](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/reinforcement_learning/index.html)
- [ROBOTIS cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)
- [LeIsaac SO-101](https://github.com/LightwheelAI/leisaac)

## BC를 첫 단계에서 뺀 이유

BC는 사람이나 scripted policy의 행동을 맞히는 지도학습입니다. PPO는 보상을 보고 행동을 탐색합니다. 두 방법을 비교하려면 입력, action, 네트워크, 학습 예산을 맞춰야 하므로 SO-101 환경 자체가 검증되지 않은 단계에서 함께 넣으면 실패 원인이 늘어납니다.

첫 결과는 순수 PPO로 고정합니다. 이후 BC를 추가할 때는 아래 중 하나를 별도 실험으로 선택합니다.

- BC policy와 PPO policy의 성공률 비교
- BC actor weight로 초기화한 PPO와 scratch PPO 비교
- demonstration을 replay buffer나 auxiliary loss로 활용하는 방법

실험을 추가하기 전에는 구현 가능성과 비교 공정성을 다시 검토합니다.

## 포트폴리오에서 피할 주장

- GUI에서 로봇 여러 대가 보인다는 이유로 대규모 학습이라고 쓰지 않습니다.
- simulation state를 사용한 정책을 vision policy로 표현하지 않습니다.
- MuJoCo 교차평가를 실물 sim-to-real 결과로 부르지 않습니다.
- tutorial 태스크를 실행한 것을 SO-101 환경 구현으로 표현하지 않습니다.
- 성공한 seed나 영상 하나만 골라 일반화 성능으로 쓰지 않습니다.
- DR을 켰다는 사실만으로 강건성이 좋아졌다고 쓰지 않습니다.

## 프로젝트가 완성됐을 때 설명할 내용

면접에서는 도구 이름보다 선택과 실패를 설명합니다.

1. 왜 BC 없이 PPO 기준선을 먼저 만들었는가?
2. 왜 그 환경 수를 선택했는가?
3. 어떤 보상이 실제 행동을 바꿨는가?
4. Isaac과 MuJoCo의 결과가 갈린 원인은 무엇이었는가?
5. 실물로 옮기려면 actor 관측과 물리 모델에서 무엇을 바꿔야 하는가?
