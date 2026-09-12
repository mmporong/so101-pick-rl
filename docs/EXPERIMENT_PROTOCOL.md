# 실험 프로토콜

## 2026-09-12 Pick & Place 확장

현재 최종 태스크는 `common/pick_place_spec.json`의 `SO101-PickPlace-v0`다.
아래 LiftCube 정의는 과거 실험 재현을 위해 보존한다. 신규 성공률은 전체 Pick & Place
성공률이며 파지 이력, 목표 위 저속 해제, 그리퍼 개방·이격 후 안정 유지가 모두 필요하다.
설계값과 실행 순서는 [Pick & Place 실행 계획](PICK_PLACE_PLAN.md)을 따른다.
평가 중 workspace 이탈·non-finite 에피소드도 분모에 포함하여 실패로 집계한다.

## 질문

이 실험은 SO-101이 Isaac에서 큐브를 들어 올리는지, 계산량을 늘렸을 때 학습 효율이 어떻게 변하는지, 학습 결과가 MuJoCo에서도 유지되는지를 확인합니다.

BC는 첫 실험에 넣지 않습니다. ROBOTIS 공개 예제에서도 PPO Lift와 10개 시연을 Mimic으로 늘려 BC를 학습하는 경로는 분리돼 있습니다. BC를 추가한다면 PPO 기준선이 끝난 뒤 별도 연구 질문과 예산을 먼저 등록합니다.

## 정책과 환경

- 알고리즘: RSL-RL PPO
- actor 입력: `common/task_spec.json`의 privileged state
- actor 출력: 6차원 joint-position delta
- 물체: rigid cube 하나
- 목표: 초기 높이보다 0.08m 이상 들어 0.5초 유지
- 학습 simulator: Isaac Sim PhysX
- 교차평가 simulator: MuJoCo

위 숫자는 v0.1.0의 사전 등록값입니다. 실측 결과가 아닙니다. 값이 물리적으로 성립하지 않으면 기존 결과를 보존하고 contract version을 올린 뒤 변경합니다.

## 실험 순서

### E0. 무학습 기준선

random policy를 ID 조건에서 평가합니다. 성공 판정이 우연이나 reset 오류로 참이 되지 않는지 확인하는 용도입니다.

### E1. 환경 수 사다리

64, 256, 512, 1024환경에서 같은 짧은 PPO 예산을 사용합니다. 각 실행에서 peak VRAM, collection steps/s, total steps/s, iteration wall time을 기록합니다. 2048환경은 1024환경보다 처리량이 좋아질 가능성이 확인된 뒤 실행합니다.

### E2. PPO 기준선

선택한 환경 수와 고정 학습 예산으로 seed 0, 1, 2를 scratch에서 실행합니다. 가장 잘 나온 seed 하나만 보고하지 않습니다.

### E3. reward ablation

기준 보상에서 한 항만 제거합니다. 첫 비교 대상은 grasp/contact term과 action-rate penalty입니다. 정책 성공률뿐 아니라 접근 거리, 큐브 높이, action 변화량을 함께 봅니다.

### E4. Domain Randomization

DR off와 on을 같은 seed와 학습 예산으로 비교합니다. 학습 범위와 held-out 범위를 겹치지 않게 저장합니다.

랜덤화 후보는 다음과 같습니다.

- 큐브 위치와 yaw
- 큐브 질량과 마찰
- 손가락 마찰
- actuator stiffness와 damping
- 관측 지연과 action 지연

한 번에 전부 켜지 않습니다. 물체 위치부터 시작하고 물리 파라미터는 one-factor 검사 뒤 묶습니다.

### E5. MuJoCo 교차평가

E2와 E4 checkpoint를 MuJoCo에서 평가합니다. Isaac과 MuJoCo의 성공률 차이를 보고하되, 차이를 바로 sim-to-real 성능으로 해석하지 않습니다.

## 평가 횟수와 지표

각 조건은 3개 seed, seed당 200에피소드입니다.

주 지표:

- lift success rate
- mean time to success
- seed 간 범위
- Isaac-MuJoCo success-rate difference

진단 지표:

- end-effector와 cube의 최소 거리
- 유효 접촉 비율
- 최대 cube height
- action-rate mean
- episode return
- peak VRAM
- collection 및 total steps/s

## 실패 단계

신규 Pick & Place 평가기는 각 실패 에피소드를 아래 하나로 분류합니다. `non_finite`, `workspace_exit`를 우선하고 나머지는 가장 높은 달성 단계 하나를 선택합니다. `transferred`는 XY와 Z 배치 허용오차를 모두 만족한 이력입니다. 안전 종료도 전체 에피소드 분모에 포함합니다.

```text
not_reached
reached_not_grasped
grasped_not_lifted
lifted_not_transferred
transferred_not_released
released_unstable
non_finite
workspace_exit
```

영상과 수치가 충돌하면 수치를 다시 계산하고 원본 trajectory를 보존합니다.

## 비교 공정성

- 대조군은 같은 task contract와 평가 seed를 사용합니다.
- ablation은 한 변수만 바꿉니다.
- 학습 예산은 iteration뿐 아니라 transition 수로도 기록합니다.
- checkpoint 선택 규칙은 평가 전에 고정합니다.
- 실패한 실행을 같은 run ID로 덮어쓰지 않습니다.
- 결과를 확인한 뒤 held-out 범위를 바꾸지 않습니다.
