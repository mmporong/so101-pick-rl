# Ubuntu 인계서: MuJoCo와 평가

## 맡은 결과

Ubuntu에서는 Isaac 학습 코드를 대신 실행하지 않습니다. SO-101 MuJoCo 환경과 평가기를 만들고 Windows에서 받은 checkpoint가 같은 관측과 행동 계약을 사용하는지 확인합니다.

## 작업 위치

```text
$HOME/so101-pick-rl
```

기존 `$HOME/so101-mobile-manipulation/sim`에는 실물과 맞춘 MuJoCo 미러가 있습니다. 원본을 수정하지 않고 source commit과 가져온 파일을 `docs/THIRD_PARTY.md`에 기록한 뒤 필요한 모델만 이 저장소의 `mujoco/assets/`로 옮깁니다.

## 시작 순서

```bash
cd "$HOME/so101-pick-rl"
git status --short --branch
python3 scripts/validate_contract.py
python3 -m unittest discover -s tests -v
```

출력된 contract SHA-256이 Windows 보고와 같아야 합니다.

## L1. MuJoCo 환경

`common/task_spec.json`을 기준으로 다음을 맞춥니다.

- 관절 이름과 순서
- radian 단위의 joint-position delta action
- action clip
- physics와 policy step 비율
- 큐브 초기 높이 기준의 성공 판정
- 에피소드 timeout

큐브는 free joint를 사용합니다. equality constraint나 코드로 손에 붙이지 않습니다.

## L2. 계약 fixture

Isaac과 MuJoCo에 같은 관절 상태와 action을 넣고 다음 값을 JSON으로 저장합니다.

- 적용 전 qpos
- clip된 action
- 적용할 joint target
- end-effector position
- cube-relative observation의 필드 순서

두 simulator의 물리가 같은 숫자를 내야 하는 것은 아닙니다. action 해석과 observation 필드 순서가 같아야 합니다.

## L3. 평가기

평가기의 입력은 checkpoint, run manifest, task contract입니다. 출력은 `common/schemas/evaluation_result.schema.json`을 따릅니다.

각 정책을 다음 조건에서 평가합니다.

| 조건 | 목적 |
| --- | --- |
| ID | 학습 분포에서 재현 여부 확인 |
| held-out | 학습에 쓰지 않은 마찰, 질량, 물체 위치 확인 |
| cross-sim | Isaac checkpoint의 MuJoCo 전이 확인 |

seed 0, 1, 2에서 각각 200에피소드를 실행합니다. 성공률이 낮아도 결과를 제외하지 않습니다.

## Ubuntu가 커밋할 범위

```text
mujoco/
evaluation/
configs/evaluation/
reports/linux/
docs/THIRD_PARTY.md
```

체크포인트와 MP4는 Git에 넣지 않습니다. 외부 artifact URI와 SHA-256만 run manifest에 남깁니다.

## 인계 결과 형식

```text
contract SHA:
MuJoCo version:
source model commit:
1환경 reset/step 결과:
fixture 정합 결과:
ID 평가:
held-out 평가:
cross-sim 평가:
실패 단계 분포:
결과 JSON 경로:
다음 시작점:
```
