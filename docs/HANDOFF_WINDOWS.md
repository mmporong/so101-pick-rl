# Windows 인계서: Isaac Lab PPO

2026-09-04 기준 G0~G4와 checkpoint 재시작을 통과했습니다. 실제 수치, source commit, checkpoint SHA-256, 남은 범위는 `reports/windows/STATUS.md`에 있습니다. 256환경 이상과 수렴 학습은 사용자가 지정한 저부하 조건 때문에 실행하지 않았습니다.

## 맡은 결과

Windows에서는 SO-101 큐브 파지 환경을 Isaac Lab 외부 extension으로 만들고 RSL-RL PPO를 실행합니다. 첫 인계의 종료점은 64환경 PPO smoke까지입니다. 장시간 본 학습은 G4 결과를 검토한 뒤 시작합니다.

## 고정 환경

| 항목 | 값 |
| --- | --- |
| 운영체제 | Windows 11 native |
| GPU | RTX 3060 12GB |
| Isaac Sim | 4.5.0, `E:\IsaacSim\isaac-sim-4.5.0` |
| Isaac Lab | v2.1.1, `%USERPROFILE%\IsaacLab` |
| Python | Isaac Sim bundled Python 3.10 |
| RL | RSL-RL 2.3.3 PPO |
| 작업 저장소 | `%USERPROFILE%\so101-pick-rl` |

Isaac Sim 5.x나 Isaac Lab 2.2+로 올리지 않습니다. 현재 ROBOTIS `cyclo_lab` 기본 환경은 Isaac Sim 5.1과 Isaac Lab 2.3이라 이 고정 환경에 직접 설치하지 않습니다. LeIsaac v0.1.2의 SO-101 자산과 LiftCube 환경을 참고하되, 원본 저장소는 수정하지 않습니다.

## 시작 전에 읽을 파일

1. `README.md`
2. `AGENTS.md`
3. `common/task_spec.json`
4. `docs/EXPERIMENT_PROTOCOL.md`
5. 이 문서

## W0. clone과 공통 계약

PowerShell에서 실행합니다.

```powershell
Set-Location "$HOME\so101-pick-rl"
git status --short --branch
python scripts\validate_contract.py
python -m unittest discover -s tests -v
```

`CONTRACT_PASS`와 contract SHA-256을 `reports/windows/w0_environment.json`에 기록합니다. Ubuntu와 SHA가 다르면 진행하지 않습니다.

## W1. 기존 스택 live check

```powershell
& "E:\IsaacSim\isaac-sim-4.5.0\python.bat" -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)"

Set-Location "$HOME\IsaacLab"
git status --short --branch
git rev-parse HEAD
& ".\isaaclab.bat" -p -c "import rsl_rl; print('RSL_RL_OK')"
& ".\isaaclab.bat" -p scripts\environments\list_envs.py
```

공식 Lift 환경을 1개만 실행해 simulator와 task registry를 확인합니다.

```powershell
& ".\isaaclab.bat" -p scripts\environments\random_agent.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 1
```

reset과 step이 확인되면 종료합니다. 이 결과는 SO-101 성공이 아니라 Windows 학습 스택의 생존 확인입니다.

## W2. 외부 extension 구성

코드는 `%USERPROFILE%\so101-pick-rl\isaaclab` 아래에 둡니다. `%USERPROFILE%\IsaacLab\source`에는 파일을 추가하지 않습니다.

구현 파일은 다음 책임으로 나눕니다.

```text
isaaclab/
├── assets/so101.py
├── tasks/lift_cube/env_cfg.py
├── tasks/lift_cube/mdp/observations.py
├── tasks/lift_cube/mdp/rewards.py
├── tasks/lift_cube/mdp/terminations.py
├── tasks/lift_cube/agents/rsl_rl_ppo_cfg.py
└── tasks/lift_cube/__init__.py
```

자산을 가져올 때 `docs/THIRD_PARTY.md`에 다음을 기록합니다.

- 원본 URL
- tag 또는 commit
- 원본 라이선스
- 복사한 파일
- 수정한 값

현재 구현은 설치형 extension 대신 저장소 내부 외부 Python 프로젝트로 동작합니다. 실행 스크립트가 `%USERPROFILE%\so101-pick-rl\isaaclab`을 `sys.path`에 추가하므로 Isaac Lab 원본이나 site-packages를 수정하지 않습니다.

```powershell
Set-Location "$HOME\so101-pick-rl"
python .\isaaclab\scripts\prepare_assets.py
```

## W3. 1환경과 64환경

태스크 ID는 `SO101-LiftCube-v0`을 사용합니다.

1환경에서는 다음을 시각적으로 확인합니다.

- base가 고정돼 있는지
- 여섯 관절 방향이 맞는지
- 손가락 collision이 큐브와 접촉하는지
- 큐브가 테이블을 통과하지 않는지
- reset마다 큐브가 지정 범위 안에서 바뀌는지

64환경에서는 headless로 다음을 검사합니다.

- 10,000 physics step 완주
- 환경별 cube pose 독립성
- non-finite state 0건
- reset 실패 0건
- peak VRAM과 steps/s

재현 명령은 다음과 같습니다.

```powershell
$python = "E:\IsaacSim\isaac-sim-4.5.0\python.bat"
& $python .\isaaclab\scripts\smoke_env.py --task SO101-LiftCube-v0 --num_envs 1 --physics_steps 1000 --action_mode zero --seed 0 --headless --output .\reports\windows\g2_so101_1env_1000.json
& $python .\isaaclab\scripts\smoke_env.py --task SO101-LiftCube-v0 --num_envs 64 --physics_steps 10000 --action_mode random --random_action_amplitude 0.1 --seed 0 --headless --output .\reports\windows\g3_so101_64env_10000.json
```

## W4. PPO smoke

보상은 한 번에 많이 넣지 않습니다.

```text
접근 거리 감소
그리퍼와 큐브 정렬
유효한 손가락 접촉
큐브 높이 상승
목표 높이 유지
action-rate penalty
```

64환경, 10 iteration으로 학습 경로만 확인합니다. 성공률 목표는 이 단계에 두지 않습니다. loss, reward, action std가 유한하고 checkpoint가 생성되면 G4를 통과합니다.

```powershell
& $python .\isaaclab\scripts\train_rsl_rl.py --task SO101-LiftCube-v0 --num_envs 64 --max_iterations 10 --save_interval 5 --run_name g4_smoke --seed 0 --headless --output .\reports\windows\g4_so101_64env_10iter.json
```

두 실행 스크립트는 시작 시 다른 작업과 겹치지 않도록 baseline GPU 사용률 40% 이하, 여유 VRAM 4GiB 이상, 시스템 CPU 사용률 70% 이하를 기본 조건으로 검사합니다. 조건을 넘으면 Isaac Sim을 띄우지 않고 `blocked_resource_guard` 보고서를 남깁니다.

## Windows가 커밋할 범위

```text
isaaclab/
configs/isaaclab/
reports/windows/
docs/THIRD_PARTY.md
```

공통 계약 변경이 필요하면 `common/`을 바로 수정하지 말고 변경 이유와 예상 영향을 commit message와 문서에 먼저 남깁니다.

## 인계 결과 형식

첫 Windows 작업이 끝나면 다음을 보고합니다.

```text
G0 contract SHA:
G1 Isaac/Lab/RSL-RL 버전:
G2 1환경 결과:
G3 64환경 10,000-step 결과:
G4 PPO smoke 결과:
peak VRAM:
total steps/s:
run manifest 경로:
실패 또는 미확인 항목:
다음 시작점:
```
