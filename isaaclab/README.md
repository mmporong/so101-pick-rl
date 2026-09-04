# Isaac Lab 영역

Windows가 소유하는 SO-101 외부 프로젝트입니다. Isaac Lab 원본에는 파일을 추가하지 않으며, 이 디렉터리를 Python 경로에 넣는 실행 스크립트가 `SO101-LiftCube-v0`을 등록합니다.

## 구현 범위

- 6개 관절의 누적 joint-position delta action, 스텝당 최대 0.05rad
- 120Hz PhysX, 30Hz 정책, 10초 에피소드
- 상태 기반 22차원 관측
- 접근·정렬·양쪽 손가락 접촉·들어 올리기·유지 보상
- 초기 큐브 높이보다 0.08m 이상을 0.5초 연속 유지하는 성공 판정
- NaN/Inf 및 workspace 이탈 종료
- RSL-RL PPO와 TensorBoard/checkpoint 검증

카메라, BC, ACT, VLA, 실물 로봇 제어는 포함하지 않습니다.

## 자산 준비

LeIsaac 릴리스 자산은 저장소에 복사하지 않습니다. 다음 명령이 고정 URL에서 USD를 받아 SHA-256을 검증한 뒤 `%LOCALAPPDATA%\so101-pick-rl\assets\leisaac`에 둡니다.

```powershell
Set-Location "$HOME\so101-pick-rl"
python .\isaaclab\scripts\prepare_assets.py
```

## 실행

모든 실행기는 시작 시 GPU 사용률, 여유 VRAM, 시스템 CPU 사용률을 확인합니다. 기본 게이트는 GPU 사용률 40% 이하, 여유 VRAM 4GiB 이상, CPU 사용률 70% 이하입니다. 조건을 넘으면 Isaac Sim을 시작하지 않고 `blocked_resource_guard` 보고서를 남깁니다.

```powershell
Set-Location "$HOME\so101-pick-rl"
$python = "E:\IsaacSim\isaac-sim-4.5.0\python.bat"

& $python .\isaaclab\scripts\smoke_env.py --task SO101-LiftCube-v0 --num_envs 1 --physics_steps 1000 --action_mode zero --seed 0 --headless --output .\reports\windows\g2_so101_1env_1000.json

& $python .\isaaclab\scripts\validate_task_semantics.py --seed 0 --headless --output .\reports\windows\task_semantics.json

& $python .\isaaclab\scripts\smoke_env.py --task SO101-LiftCube-v0 --num_envs 64 --physics_steps 10000 --action_mode random --random_action_amplitude 0.1 --seed 0 --headless --output .\reports\windows\g3_so101_64env_10000.json

& $python .\isaaclab\scripts\train_rsl_rl.py --task SO101-LiftCube-v0 --num_envs 64 --max_iterations 10 --save_interval 5 --run_name g4_smoke --seed 0 --headless --output .\reports\windows\g4_so101_64env_10iter.json
```

학습 로그와 checkpoint는 `isaaclab/logs/`에 남고 Git에서는 제외됩니다. 작은 JSON 보고서만 `reports/windows/`에 커밋합니다.

현재 Windows G0~G4 결과와 checkpoint SHA-256은 `reports/windows/STATUS.md`를 봅니다.
