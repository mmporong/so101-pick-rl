# Windows Isaac Lab 검증 상태

측정일은 2026-09-04, 브랜치는 `feat/isaaclab-windows`입니다. Isaac Sim 4.5.0과 Isaac Lab v2.1.1 원본은 수정하거나 업그레이드하지 않았습니다. 실물 SO-101도 움직이지 않았습니다.

## 게이트 결과

| 게이트 | 상태 | 실제 증거 |
| --- | --- | --- |
| G0 | PASS | contract SHA-256 `61795eb3e46695a8ef837266003d92665db0fc424ed29dc976b1d14f22bf6750`, unittest 15개 PASS |
| G1 | PASS | 공식 `Isaac-Lift-Cube-Franka-v0`, 1환경 20 physics step, simulator error 0건 |
| 의미 계약 | PASS | raw action 10을 0.05rad로 clip, 실제 previous action 일치, 부분 reset, 양쪽 contact sensor, 15-step success hold 확인 |
| G2 | PASS | SO-101 1환경 1,000 physics step, NaN/Inf·관절 limit·테이블 관통·simulator error 0건 |
| G3 | PASS | SO-101 64환경 10,000 physics step, timeout reset 512회, 비정상 종료 0건 |
| G4 | PASS | 64환경 PPO 10 iteration, 15,360 transitions, 필수 scalar 각 10개, checkpoint 3개 |
| 재시작 | PASS | `model_9.pt`에서 optimizer와 policy를 읽어 iteration 10의 `model_10.pt` 생성 |
| 256~2,048환경 | NOT RUN | 공유 자원 보호를 위해 실행하지 않음 |
| 수렴 학습·성공률 | NOT RUN | G4는 실행 경로 smoke이며 정책 수렴 판정이 아님 |

## 실행 환경

| 항목 | 값 |
| --- | --- |
| Windows | Windows 11 Pro 25H2, build 26200 |
| GPU | NVIDIA GeForce RTX 3060 12GB, driver 610.62 |
| CPU | AMD Ryzen 7 5800X, 8 core / 16 logical core |
| Isaac Sim | `4.5.0-rc.36+release.19112.f59b3005.gl` |
| Isaac Lab | v2.1.1, commit `90b79bb2d44feb8d833f260f2bf37da3487180ba` |
| Python | 3.10.15 |
| PyTorch/CUDA | 2.7.0+cu128 / CUDA 12.8 |
| RSL-RL | 2.3.3 |

## G2·G3 측정

| 항목 | G2 | G3 |
| --- | ---: | ---: |
| 환경 수 | 1 | 64 |
| physics step | 1,000 | 10,000 |
| wall clock | 12.98초 | 144.26초 |
| physics frame/s | 77.03 | 69.32 |
| aggregate sim step/s | 77.03 | 4,436.50 |
| 최대 VRAM | 2,809MiB | 2,826MiB |
| 평균/최대 GPU | 23.23% / 38% | 33.14% / 41% |
| 평균/최대 system CPU | 37.34% / 54.3% | 49.99% / 80.1% |
| simulator warning/error | 18 / 0 | 18 / 0 |

18개 warning은 crash reporter 부재, repo의 선택적 `rendering_modes` 설정 부재, MaterialX/OmniHub, Isaac Sim 4.5 deprecated dynamic control, USD PreviewSurface discovery 메시지입니다. 보고서의 Kit log 경로와 warning sample에 보존했으며 태스크·PhysX error는 없었습니다.

64환경 G3의 평균 부하는 제한 범위였지만 system CPU가 순간 80.1%까지 올라갔습니다. 이 때문에 이번 작업에서는 256환경 이상 scale ladder를 실행하지 않았고, 검증된 최대 안정 병렬 환경 수는 64입니다.

## PPO smoke와 산출물

- 실행 로그: `isaaclab/logs/rsl_rl/so101_lift_cube/2026-09-04_16-39-32_g4_final/`
- TensorBoard event: `isaaclab/logs/rsl_rl/so101_lift_cube/2026-09-04_16-39-32_g4_final/events.out.tfevents.1788507575.DESKTOP-991BNBV.31544.0`
- 최종 smoke checkpoint: `isaaclab/logs/rsl_rl/so101_lift_cube/2026-09-04_16-39-32_g4_final/model_9.pt`
- `model_9.pt` SHA-256: `54f4a74124a5b34e44d77e43b9e01dd39f63cb7baa6a36edaeabfaf765544aa5`
- 재시작 checkpoint: `isaaclab/logs/rsl_rl/so101_lift_cube/2026-09-04_16-40-49_g4_resume_final/model_10.pt`
- `model_10.pt` SHA-256: `33caefdff8c8fc4e7421b97fb44f2e73503b58eaf7d7cd92c7037b467861e14e`
- 10-iteration wall clock: 19.26초
- 평균/최대 GPU: 24.03% / 42%
- 최대 VRAM: 2,835MiB
- 평균/최대 system CPU: 39.93% / 69.2%
- 마지막 `Train/mean_reward`: 2.3834
- 마지막 contact reward: 0.00167
- lift success: 0

checkpoint와 TensorBoard 파일은 로컬 경로에 있으며 Git에서는 제외됩니다. JSON 보고서와 manifest가 경로, SHA-256, source commit을 기록합니다. 10 iteration 안에 접촉이 발생하기 시작했지만 들어 올리기 성공은 없으므로 “학습 파이프라인이 동작한다”까지만 주장할 수 있습니다.

## 정본 보고서

- `reports/windows/w0_environment.json`
- `reports/windows/g1_franka_smoke.json`
- `reports/windows/task_semantics_final.json`
- `reports/windows/g2_so101_1env_1000_final.json`
- `reports/windows/g3_so101_64env_10000_final.json`
- `reports/windows/g4_so101_64env_10iter_final.json`
- `reports/windows/g4_so101_64env_resume_final.json`

각 JSON 옆의 `_manifest.json`은 `isaaclab/scripts/validate_run_manifest.py`로 검증합니다. 정본 실행은 모두 `git_dirty=false`인 source commit에서 시작했습니다.

정본 실행 후 검증 하네스를 추가로 강화했습니다. 최신 코드는 G3가 `non_finite=0`, `workspace_exit=0`, 중복 제거한 자동 reset 100회 이상을 만족해야만 PASS로 판정합니다. 실제 reset 환경마다 반환 관측, episode length, 로봇·큐브 상태, 에피소드 초기 높이를 검사해 실패 수와 사유도 기록합니다. resource sample이나 이번 명령에 결합된 Kit log가 없으면 성공을 허용하지 않습니다. 판정식과 종료 플래그 중복 제거는 순수 회귀 테스트에 포함했습니다. 기존 G3 정본은 timeout 512회를 기록했지만 reset 후 상태 검사가 도입되기 전 측정이므로, 다른 GPU 작업 종료 후 최신 하네스로 다시 측정합니다.

## 재현 명령

```powershell
Set-Location "$HOME\so101-pick-rl"
$python = "E:\IsaacSim\isaac-sim-4.5.0\python.bat"

python .\scripts\validate_contract.py
python -m unittest discover -s tests -v
& $python .\isaaclab\scripts\validate_task_semantics.py --seed 0 --headless --output .\reports\windows\task_semantics_rerun.json
& $python .\isaaclab\scripts\smoke_env.py --task SO101-LiftCube-v0 --num_envs 1 --physics_steps 1000 --action_mode zero --seed 0 --headless --output .\reports\windows\g2_rerun.json
& $python .\isaaclab\scripts\smoke_env.py --task SO101-LiftCube-v0 --num_envs 64 --physics_steps 10000 --action_mode random --random_action_amplitude 0.1 --seed 0 --headless --output .\reports\windows\g3_rerun.json
& $python .\isaaclab\scripts\train_rsl_rl.py --task SO101-LiftCube-v0 --num_envs 64 --max_iterations 10 --save_interval 5 --run_name g4_rerun --seed 0 --headless --output .\reports\windows\g4_rerun.json
```

각 Isaac 실행은 시작 시 baseline GPU 40% 이하, 여유 VRAM 4GiB 이상, system CPU 70% 이하를 검사합니다. 조건을 넘으면 시뮬레이터를 시작하지 않고 `blocked_resource_guard` 보고서를 남깁니다.
