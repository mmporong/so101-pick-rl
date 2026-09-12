# 외부 자산과 코드 출처

## 사용 예정 자료

| 자료 | 출처 | 고정 버전 | 현재 상태 | 사용 범위 |
| --- | --- | --- | --- | --- |
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | v2.1.1, commit `90b79bb2d44feb8d833f260f2bf37da3487180ba` | adapted | BSD-3-Clause, manager-based 환경 API와 `scripts/reinforcement_learning/rsl_rl/train.py` 실행 구조 참고 |
| RSL-RL | https://github.com/leggedrobotics/rsl_rl | 2.3.3 | Windows 설치 확인 대상 | PPO |
| LeIsaac 소스 | https://github.com/LightwheelAI/leisaac | v0.1.2, commit `47df88fee80b1d8b27faf1632ee500c63c4cf1c7` | adapted | `source/leisaac/leisaac/assets/robots/lerobot.py`의 관절명·액추에이터 초기값과 LiftCube 링크/좌표 구조 참고 |
| LeIsaac SO-101 USD | https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd | SHA-256 `64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58` | runtime_download | 로컬 캐시에서 SO-101 articulation 로드 |
| ROBOTIS cyclo_lab | https://github.com/ROBOTIS-GIT/cyclo_lab | main 조사 시점 | reference_only | RL과 IL 공개 워크플로 비교 |
| 기존 SO-101 MuJoCo 미러 | `$HOME/so101-mobile-manipulation/sim` | 가져오는 시점의 Git commit 기록 필요 | reference_only | MuJoCo 모델과 좌표계 참고 |

`reference_only`는 아직 파일을 복사하거나 설치 결과를 이 저장소의 구현으로 사용하지 않았다는 뜻입니다.

LeIsaac 소스 저장소의 라이선스는 Apache-2.0입니다. SO-101 USD 릴리스 자산에는 별도의 라이선스 문구가 확인되지 않아 바이너리를 이 저장소에서 재배포하지 않습니다. `isaaclab/scripts/prepare_assets.py`가 원본 릴리스 URL에서 사용자 로컬 캐시로 직접 내려받고 고정 SHA-256을 검증합니다.

LeIsaac LiftCube 코드는 카메라·텔레오퍼레이션/Mimic 예제이며 보상 항이 비어 있고, 성공 기준도 이 프로젝트 계약과 다릅니다. 따라서 환경 코드는 복사하지 않았습니다. 이 프로젝트는 Isaac Lab 2.1.1의 manager-based API로 PPO 관측·보상·종료·실행기를 별도 작성했습니다.

| 원본 | 이 저장소 파일 | 변경 내용 |
| --- | --- | --- |
| LeIsaac `source/leisaac/leisaac/assets/robots/lerobot.py` | `isaaclab/so101_pick_rl/assets/so101.py` | 관절명·STS3215 actuator 초기값을 가져오고 로컬 캐시 경로, contact sensor 활성화, 안정성 설정을 추가 |
| Isaac Lab `scripts/reinforcement_learning/rsl_rl/train.py` | `isaaclab/scripts/train_rsl_rl.py` | 외부 태스크 등록, 자원 사전 게이트, TensorBoard/체크포인트 유한값 검사, manifest 생성을 추가 |
| Isaac Lab `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` | `isaaclab/so101_pick_rl/tasks/pick_place/environment.py` | 같은 v2.1.1 commit의 step 루프를 유지하고 매 scene.update 직후 120Hz 파지/배치 이력 및 성공 latch를 추가. 상류 갱신 때 재대조 필요. BSD 고지는 `docs/licenses/ISAACLAB-BSD-3-Clause.txt`에 보존 |

## 파일을 가져올 때 기록할 항목

외부 파일을 복사하거나 수정하면 아래 내용을 같은 표에 추가합니다.

- 원본 파일 경로
- 원본 commit
- 복사한 저장소 경로
- 라이선스와 고지 의무
- 원본 대비 변경 내용
- 변환 또는 생성 명령
- 결과 파일 SHA-256

라이선스를 확인하지 못한 mesh, texture, USD는 공개 저장소에 넣지 않습니다.
