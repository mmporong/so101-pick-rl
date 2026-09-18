# AGENTS.md

이 저장소는 SO-101 단일 팔의 큐브 파지를 Isaac Lab에서 학습하고 MuJoCo에서 교차평가하는 프로젝트다. 상위 전역 규칙을 따르며, 아래 규칙이 이 저장소의 작업 경계를 정한다.

## 프로젝트 경계

- 기본 목표는 카메라 없는 상태 기반 PPO다. 2026-09-18 사용자 승인에 따라 기존
  LeIsaac 시연을 활용하는 state actor BC 초기화와 병렬 PPO를 별도 실험으로 허용한다.
  기존 4cm/delta 과제는 보존하고, 원본 3cm/절대 목표각 과제는 별도 계약으로 검증한다.
  ACT, VLA와 실물 제어는 이번 학습 범위에 넣지 않는다.
- Windows는 Isaac Sim 4.5.0, Isaac Lab 2.1.1, RSL-RL 2.3.3으로 학습한다.
- Ubuntu는 MuJoCo 환경, 평가기, 결과 비교를 담당한다.
- Isaac과 MuJoCo 코드는 이 저장소에 함께 둔다. 실물 SO-101와 IL 코드는 `$HOME/so101-mobile-manipulation`에 남긴다.
- 실물 팔을 움직이는 명령을 이 저장소에 추가하거나 실행하지 않는다.

## 소유 경로

- Windows 작업: `isaaclab/`, `configs/isaaclab/`, `reports/windows/`
- Ubuntu 작업: `mujoco/`, `evaluation/`, `reports/linux/`
- 공통 계약: `common/`, `configs/evaluation/`, `docs/`
- 공통 계약은 한쪽에서 변경하고 커밋한 뒤 다른 PC가 pull한다. 두 PC에서 동시에 고치지 않는다.

## 구현 원칙

- Isaac Lab 본체와 LeIsaac 원본을 직접 수정하지 않는다. 이 저장소를 외부 extension으로 유지한다.
- SO-101 자산과 환경은 출처 commit, 라이선스, 변경 내역을 기록한 뒤 가져온다.
- 관측 순서, 행동 단위, 성공 판정은 `common/task_spec.json`을 정본으로 사용한다.
- 각도는 radian, 길이는 meter, 시간은 second로 저장한다. 화면 표시에서만 degree나 millimeter로 바꾼다.
- 큐브를 손에 강제로 붙이는 로직은 학습과 평가에 사용하지 않는다. PhysX 또는 MuJoCo 접촉으로 파지가 성립해야 한다.
- 튜토리얼 태스크를 실행한 결과와 SO-101 커스텀 태스크 결과를 섞지 않는다.

## 실행 증거

모든 학습 실행은 다음을 남긴다.

- Git commit과 dirty 여부
- task contract SHA-256
- simulator, library, GPU, seed, 환경 수
- 학습 iteration, transition 수, wall-clock time
- peak VRAM과 simulation steps/s
- checkpoint SHA-256
- ID 평가와 held-out 평가 결과

원본 체크포인트, 데이터셋, TensorBoard 로그, MP4는 Git에 넣지 않는다. `reports/`에는 작은 JSON, CSV, PNG와 외부 artifact의 URI 및 SHA-256만 둔다.

## 완료 주장

- 실행 전 항목은 `not_run`, 실패한 항목은 `failed`, 중단된 실행은 `interrupted`로 기록한다.
- 성공 영상 하나로 정책 성공을 주장하지 않는다. 사전에 고정한 seed와 에피소드 수로 평가한다.
- 시뮬레이션 성공을 실물 성공으로 표현하지 않는다.
- Domain Randomization의 효과가 없거나 나빠져도 결과를 보존한다. 결과를 보고 기준을 낮추지 않는다.
- 문서, 코드, 설정, 로그가 같은 commit과 contract SHA를 가리킬 때만 포트폴리오 결과로 사용한다.

## 검증 순서

1. `python scripts/validate_contract.py`
2. `python -m unittest discover -s tests -v`
3. 대상 simulator의 1환경 reset/step
4. 64환경 smoke
5. 학습 및 평가
