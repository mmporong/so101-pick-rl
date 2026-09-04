# SO-101 pick RL 통합 인계서

## 프로젝트 상태

저장소 계약과 실행 계획은 작성됐습니다. PPO 학습, SO-101 Isaac task, MuJoCo 교차평가는 아직 실행하지 않았습니다.

현재 contract SHA-256은 각 PC에서 `python scripts/validate_contract.py`로 다시 계산합니다. 문서에 복사된 과거 SHA를 신뢰하지 않습니다.

## 새 세션에 전달할 지시

```text
이 저장소는 SO-101 큐브 파지를 Isaac Lab의 RSL-RL PPO로 학습하고 MuJoCo에서 교차평가하는 프로젝트다.

먼저 AGENTS.md, common/task_spec.json, docs/PROJECT_PLAN.md를 읽고 python scripts/validate_contract.py와 python -m unittest discover -s tests -v를 실행한다.

Windows 작업이면 docs/HANDOFF_WINDOWS.md의 G0-G4만 수행한다. 기존 %USERPROFILE%\IsaacLab과 E:\IsaacSim\isaac-sim-4.5.0은 읽기 전용으로 사용하고, 새 코드는 이 저장소의 isaaclab/ 아래에만 작성한다. feat/isaaclab-windows 브랜치를 사용한다.

Ubuntu 작업이면 docs/HANDOFF_LINUX.md의 L1-L3을 수행한다. 새 코드는 mujoco/, evaluation/ 아래에 작성하고 feat/mujoco-linux 브랜치를 사용한다.

실물 로봇을 움직이지 않는다. BC, ACT, VLA를 추가하지 않는다. 실행하지 않은 결과를 성공으로 쓰지 않는다. checkpoint와 대용량 로그는 Git에 넣지 않는다. 각 관문의 검사 결과와 실패 원인을 reports/ 아래의 작은 JSON 또는 CSV로 남긴다.
```

## 파일 소유권

| 작업 | 브랜치 | 수정 경로 |
| --- | --- | --- |
| Windows Isaac/PPO | `feat/isaaclab-windows` | `isaaclab/`, `configs/isaaclab/`, `reports/windows/` |
| Ubuntu MuJoCo/평가 | `feat/mujoco-linux` | `mujoco/`, `evaluation/`, `configs/evaluation/`, `reports/linux/` |
| 통합 | `main` | 검증을 통과한 두 브랜치 병합과 포트폴리오 결과 정리 |

`common/`은 한 PC에서 변경하고 commit한 뒤 다른 PC가 pull합니다. 두 브랜치가 공통 계약을 각각 바꾸지 않습니다.

## 첫 번째 종료점

Windows 인계의 첫 종료점은 다음 다섯 가지입니다.

1. 두 공통 검증 명령 통과
2. Isaac Sim, Isaac Lab, RSL-RL live check
3. SO-101 1환경 reset/step
4. SO-101 64환경 10,000 physics-step smoke
5. 64환경 PPO 10-iteration smoke와 run manifest

이는 정책 학습 성공이 아니라 학습 경로가 열린 상태입니다. 본 학습은 `docs/PROJECT_PLAN.md`의 G5부터 이어갑니다.
