# SO-101 pick RL 통합 인계서

## 프로젝트 상태

2026-09-12에 최종 목표가 Pick & Place 전체로 확장됐다. 현재 작업은
`docs/PICK_PLACE_PLAN.md`와 `common/pick_place_spec.json` revision 2를 따른다.
현재 증거 정본은 `reports/windows/PICK_PLACE_STATUS.md`다.
1,024환경 GPU PPO 300 iteration과 3 seed × 200회 독립 ID 평가를 완료했지만,
전체 성공은 0/600이다. 525회는 집고 들어 올린 뒤 목표 배치에 실패했다.
배치/해제 학습은 미완료이며, 실행 기반 검증 통과를 정책 성공으로 혼동하지 않는다.
LiftCube 및 revision 1 체크포인트를 revision 2에 그대로 resume하지 않는다.

`reports/windows/STATUS.md`는 기존 LiftCube 기록이다. 신규 Pick & Place는 64/256/512/1,024환경 확장과 수정된 계약의 G2/G3/G4·재시작까지 검증했다. MuJoCo·held-out·DR/ablation은 미실행이다.

현재 contract SHA-256은 각 PC에서 `python scripts/validate_contract.py`로 다시 계산합니다. 문서에 복사된 과거 SHA를 신뢰하지 않습니다.

## 2026-09-13 시네마틱 재촬영 완료 (최신)

탑뷰 대신 약 37° 사선 전체 화면에서 시작해, 첫 4초의 실제 진행 이력으로 고른
4×4 환경 구역에 10초 동안 부드럽게 접근하고 이후 약하게 선회하도록 재촬영했다.
근접 화면은 약 12대가 온전히 보이고 가장자리 5–6대가 일부 보인다.
선택 당시 16환경 중 15환경에서 유효한 들기 이력을 관찰했지만 전체 성공은 0이다.
이는 잘 보이는 구역을 고른 편향된 촬영이며 정책 성공률 평가가 아니다.

clean source `9d3003275d5a3f3638c0253ba0f63e4ab46d64ca`에서 1,024환경·cuda:0으로
실제 PPO 50 update(iteration 340–389)를 수행했다. 원본 `model_339.pt`를 보존하고
별도 로그에 `model_389.pt`를 저장했다. 계약·보상·물체 크기는 변경하지 않았다.
영상은 `artifacts/cinematic_20260913_live1024/parallel_training.mp4`에 있으며
1920×1080·30fps·1,200프레임·40초다. MP4·체크포인트는 로컬에만 보관한다.
[시네마틱 촬영 보고서](reports/windows/pp_cinematic_video_20260913.json)에 경로와 해시를 기록했다.

CPU 75개·계약·컴파일·실행 보고서·영상 전체 디코드·프레임/해시 검증 PASS.
코드 리뷰 COMMENT(독립 Python 정적 분석의 Isaac SDK 환경 한계), 아키텍처와 사선 구도 검토 CLEAR.
14초·24초·마지막 프레임까지 확인했다. model_389의 정식 ID/held-out 평가는 미실행이다.
영상은 optimizer 대기 시간을 생략한 시뮬레이션 시간이며, reset 전 종료 상태를 보장하는
전이 정렬 학습 데이터셋이 아니다. 코드와 작은 기록만 사용자 요청에 따라 push한다.

## 2026-09-13 병렬 학습 영상 촬영 완료 (이전 영상)

사용자가 실제 PPO 업데이트를 포함한 짧은 재개 촬영을 선택했다. clean source
`e40bc5c069e012b7aed1eafd1aca774b1433bbb6`에서 원본 `model_299.pt`를 보존하고
1,024환경·40 update(iteration 300–339)를 새 로그 디렉터리에서 실행했다.
1280×720·30fps·960프레임·32초 영상과 별도 `model_339.pt`를 검증했다.
실행 증거·해시·경로는 [촬영 보고서](reports/windows/pp_parallel_video_20260913.json)에 있다.
영상은 `artifacts/parallel_training_20260913_live1024/parallel_training.mp4`에만 보존하며
사용자 지시대로 Git에 넣거나 업로드하지 않는다.

렌더러는 `--headless --enable_cameras --device cuda:0 --kit_args=--/app/vulkan=false`로
D3D12 경로를 사용했다. Isaac Sim/Lab 설치본은 수정하지 않았다.
CPU 71개·계약·컴파일·영상 전체 디코드 및 프레임/해시 검증 PASS.
코드 리뷰 COMMENT(정적 분석 도구 부재), 아키텍처 CLEAR이며 코드 차단 이슈는 없다.
단일 환경 진단은 600전이·601프레임과 reset 전 마지막 상태 보존을 확인했다.

이번 실행은 촬영용 기준선 연장이다. **3cm·보상·커리큘럼 개선은 아직 하지 않았고,
새 모델의 정식 성공률도 평가하지 않았다.** 원본의 0/600을 새 모델의 수치로 옮기지 않는다.
아래 실패 계측·개선 순서는 그대로 남는다. 이후 같은 구도로 개선 전후 비교 영상을 찍되,
정책 성공 주장은 별도 고정 seed 평가로 검증한다.

## 2026-09-12 재개 메모: 개선 조사 완료, 구현·재학습 전(당시 기록)

당시 사용자 요청은 "나중에 진행하게 내용 정리"였다. 당시에는 인계 기록만 남기고 새 학습을 시작하지 않았다. 이후 허용된 촬영용 학습 재개는 위 2026-09-13 기록과 구분한다.

### 중단 시점과 실행 방식

- 3cm 큐브, 보상 전환, 궤적 계측은 **아직 구현하지 않았다**. 큐브는 여전히 4cm·50g이다.
- 일반 구현 단계에서 최초 읽기 전용 명령이 사용자에 의해 중단됐다. 정리 시작 시 worktree는 clean이었고, 이 프로젝트의 학습·평가 명령을 실행 중인 `kit.exe`/`python.exe`/`pythonw.exe` 프로세스는 조회되지 않았다. 프로세스 상태는 재개 시 다시 확인한다.
- 앞서 제안한 `ralplan`은 설치된 스킬의 공식 host receipt 검증 기능 부재로 실행 전환이 막혔다. 사용자는 이후 **ralplan을 제외한 일반 구현·검증 방식**으로 진행하는 데 동의했다. 다음 세션은 같은 합의계획 절차를 자동 재호출하지 말고, 현재 호스트·상위 지침 안에서 일반 구현 경로를 사용한다. 사용자 동의를 host receipt로 해석하거나 기존 Ralplan gate를 승인됐다고 기록하지 않는다.
- 원격 push·실물 실행 권한은 없다. 로컬 변경은 검증·독립 리뷰·경로별 스테이징·로컬 커밋까지 보존한다.

### 보존된 기준선

- 브랜치: `feat/isaaclab-windows`.
- 학습 source: `a678ceda7dad1b9bc6d1c50c28a663fcc83fdb21`. 결과 기록 commit: `7520e13`.
- Isaac Sim 4.5.0 / Isaac Lab 2.1.1 / RSL-RL 2.3.3, RTX 3060 12GB.
- 실제 환경·정책 `cuda:0`, 1,024환경, 300 PPO iteration, 7,372,800 transitions. 학습 716.6068초. 이 분량은 초기 기준선이지 수렴 판정 기준이 아니다.
- **학습 seed는 0 하나**다. 평가 seed 0/1/2 각 200회와 구분한다.
- 평가: 전체 성공 0/600, 접촉 후 들기 미달 71, 유효하게 든 뒤 목표 배치 미달 525, 영역 이탈 4.
- 체크포인트: `isaaclab/logs/rsl_rl/so101_pick_place/2026-09-12_13-48-21_pp_r2_baseline300/model_299.pt`.
- 체크포인트 SHA-256: `d8d87329b4eeb3f2b45997aa3b61fc5eff6e411e9d2500b2711927ab4d26b35e`.
- 계약 revision 2 SHA-256: `f4af446ce9644e558653da595d34b0c752dde974e3951051dee7d8d672dc52d3`.
- TensorBoard·agent.yaml·run_contract.json은 같은 로그 디렉터리에 있다. 체크포인트와 큰 로그는 로컬 Git 비추적 자료다. 다른 기기에서는 존재 여부부터 확인하고 Git clone만으로 복구된다고 가정하지 않는다.
- 이전 검증: CPU 58개, CUDA 상태·계약 테스트 33개 PASS. 이번 문서 정리에서 재학습·CUDA 재검증을 수행한 것은 아니다.

### 조사 결과: 확인된 사실과 가설

1. 원본 TensorBoard의 50-iteration 구간 평균 운반 보상은 `0 → 0.013 → 0.088 → 0.408 → 1.198 → 1.949`로 증가했다. 마지막 50회 배치 보상 평균은 약 0.00216이며, 놓기·안정 유지·성공 보상은 전체 300회에서 0이다. 값은 학습 로그 보상이지 성공률이 아니다. **정체·수렴 또는 학습량 충분을 단정하지 않는다.**
2. `lifted_not_transferred`는 가장 높은 도달 단계의 분류다. 유효한 들기 이력은 있지만, 유효한 운반 상태에서 목표 XYZ 조건에 도달한 이력이 없다는 뜻이다. XY 미도달, 목표 위 Z 하강 실패, 중도 접촉 상실을 현재 집계만으로 구분할 수 없다. "그리퍼가 열리지 않는 것이 원인"도 아직 미확인이다.
3. `isaaclab/so101_pick_rl/tasks/pick_place/mdp.py` 및 상속된 Lift 보상에서 접근·정렬·접촉·들기 보상이 post-lift에도 남는다. transport는 `carry_valid`와 XY 거리, placement는 `carry_valid`와 XYZ 거리로 보상한다. release 보상은 실제 release 상태가 된 뒤 시작된다. **들고 머무는 유인과 다음 단계 탐색 부족은 유력한 가설이지 검증된 정책 실패 원인이 아니다.**
4. `isaaclab/so101_pick_rl/tasks/lift_cube/mdp/actions.py`는 내부 `_target`에 관절 delta를 누적하지만, 관측에는 실제 관절 위치·속도와 마지막 delta만 있고 누적 target은 직접 포함되지 않는다. 제어 지연·포화 시 부분관측 후보이므로 목표–실제 관절 오차를 먼저 측정한다. 큐브 자세 관측도 후순위 검토 대상이다.
5. 현재 4cm·50g은 Lift 환경 코드에 하드코딩돼 있고 Pick & Place 계약 해시는 JSON을 대상으로 한다. 큐브 상수만 바꾸면 geometry 변경이 계약 해시에 반영되지 않는다. 3cm 실험 전 크기·질량을 명시적인 실험 계약에 포함해야 한다.
6. revision 1의 정상 하강 판정 버그는 revision 2에서 수정됐다. 현재 결과를 과거 버그로 설명하지 않는다. `gamma=0.99`/30Hz에서 10초 뒤 보상 가중치는 약 0.049지만, 할인율이 실패 원인이라는 증거는 없다. 변경한다면 보상 상한·성공 보너스 검증과 함께 별도 비교한다.

### 재개할 작업 순서

1. **기준선 상태 확인:** cwd·branch·git status, 체크포인트/계약 SHA, GPU 작업 점유를 확인한다. 기존 G0–G4를 무조건 반복하지 말고 변경 범위에 맞춰 검증한다. Isaac Lab/Sim 본체는 수정하지 않는다.
2. **실패 경로 계측부터 구현:** `isaaclab/scripts/evaluate_policy.py`, `isaaclab/so101_pick_rl/policy_evaluation.py`, Pick & Place 환경/상태 연동부에 에피소드별 최소 XY 오차, 목표 XY 영역 진입 여부, 영역 안 Z 오차, 접촉력·큐브 속도·그리퍼 개방률, `carry_valid` 해제 시점/사유, `release_ready`/`released`, 관절 목표–실제 오차·한계 포화를 기록한다. XY/Z 최소값을 서로 다른 시점에서 따로 만족했다고 목표 도달로 합치지 않는다. 기존 결과 JSON은 덮어쓰지 않는다.
3. **고정 체크포인트 진단:** 먼저 소수 에피소드로 계측과 부분 reset 격리를 검증하고, 기존 r2 `model_299.pt`의 실패를 XY 접근/하강/운반 이력 상실로 구분한다. 관측·행동·보상·성공 조건은 계측만을 위해 바꾸지 않는다. 예전 체크포인트 호환 검사도 우회하지 않는다.
4. **연장 기준선과 보상 전환 비교:** 증가 중이던 운반 학습을 고려해 기존 보상의 학습 연장도 대조군으로 둔다. 개선군은 들기 이후 기존 접근·들기 보상의 비중을 조절하고 운반→하강→놓기/후퇴 신호를 분리하는 후보를 검토한다. 공통 시작점·총 전이 수·평가 분포를 고정하고, 새 계약으로의 전이는 명시적으로 기록한다. 변경된 보상에 이전 optimizer/critic을 그대로 재개할지 포함해 실험 조건을 고정하며 호환 검사를 무력화하지 않는다.
5. **거리 커리큘럼은 별도 비교:** 실제 접촉 파지·8cm 들기·제어된 해제·안정 유지 조건을 지키면서 가까운 목표부터 시작해 원래 분포로 확장한다. 시작 거리 6cm 미만을 사용하는 쉬운 학습 분포는 기존 최종 평가 계약과 분리한다. 최종 성공률은 원래 분포에서만 주장한다. 큐브 attachment나 합성 파지 이력을 학습·정책 평가에 사용하지 않는다.
6. **3cm 비교 실험:** 우선 4cm/50g 대 3cm/50g을 동일한 절대 성공 허용오차·목표 분포·보상·학습 예산으로 비교한다. 3cm의 초기 중심은 0.017m, 배치 중심은 0.015m로 맞추고 실제 collision·접촉을 검증한다. 동일 밀도 조건은 `50 × (3/4)^3 = 21.09375g`의 별도 비교군이며 크기와 질량을 함께 바꾸는 실험이다. 같은 질량이어도 관성이 바뀌므로 "기하만 완벽히 분리"했다고 표현하지 않는다. 기존 LiftCube 설정/결과는 보존하고 PP 전용 계약으로 분리한다.
7. **중간 평가와 완료 구분:** 여러 변경을 한 번에 넣지 않는다. 학습 전에 비교 예산·체크포인트 평가 간격·단계별 통과 기준을 고정한다. 300회 종료나 총보상 상승을 완료로 삼지 않는다. 유망한 조건은 복수 학습 seed로 반복하고, 최종 ID 평가는 seed 0/1/2 각 200회로 모든 실패를 분모에 포함한다. 전체 Pick & Place 성공이 없으면 개선 완료라고 하지 않는다. held-out/DR/ablation/MuJoCo/영상 검증 여부는 별도 표시한다.

### 조사에 사용한 외부 근거와 적용 한계

- [Isaac Lab v2.1.1 Lift 보상](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.1.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py): 높이 조건의 목표 추적 구현. 완전한 release/stability 예제는 아니다.
- [Ng 등의 potential-based shaping 논문](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf): `gamma * Phi(next) - Phi(current)`와 정책 보존 조건. 임의 단계 보상이나 시간에 따라 바뀌는 curriculum 전체에 보존 정리를 자동 적용하지 않는다.
- [Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html): 쉬운 시작 조건에서 성공 범위를 확장하는 연구 근거. 목표 거리 curriculum은 이 프로젝트에 대한 적용 후보이며 SO-101 성공 보장은 아니다.
- [Fetch Pick And Place 공식 문서](https://robotics.farama.org/main/envs/fetch/pick_and_place/): 목표 거리 기준의 과제로, 현재의 해제·손 8cm 후퇴·1초 안정 요구와 성능을 직접 비교할 수 없다.
- [PhysX 접촉 설정 문서](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html): contact/rest offset과 시간 간격의 관계를 설명하는 일반 근거. 문서 버전을 현재 Isaac Sim 번들 PhysX 버전이라고 간주하지 않는다.

## 새 세션에 전달할 지시

```text
이 저장소는 SO-101 큐브 파지를 Isaac Lab의 RSL-RL PPO로 학습하고 MuJoCo에서 교차평가하는 프로젝트다.

먼저 HANDOFF.md의 "2026-09-12 재개 메모", AGENTS.md, common/pick_place_spec.json, reports/windows/PICK_PLACE_STATUS.md, docs/PICK_PLACE_PLAN.md를 읽고 현재 cwd와 git status를 확인한다. 사용자가 구현 재개를 요청한 경우 python scripts/validate_contract.py와 python -m unittest discover -s tests -v를 실행한다. 정리 요청만 받은 상태에서는 새 학습을 시작하지 않는다.

현재 4cm·50g 기준선은 전체 성공 0/600이며, 3cm 변경·보상 개선·상세 궤적 계측은 아직 구현 전이다. 사용자는 ralplan을 제외한 일반 구현·검증 방식에 동의했다. 실패 경로 계측부터 진행하고 기존 정책의 XY 접근/Z 하강/운반 이력 상실을 구분한 뒤, 연장 기준선·보상 전환·거리 curriculum·3cm 조건을 한 번에 섞지 않고 비교한다. 목표는 들기만이 아니라 배치·해제·안정 유지까지다.

Windows 작업이면 기존 G0-G4를 무조건 처음부터 반복하지 말고, revision 2 checkpoint와 평가의 배치/해제 병목을 확인한다. 기존 %USERPROFILE%\IsaacLab과 E:\IsaacSim\isaac-sim-4.5.0은 읽기 전용으로 사용한다. 구현은 이 저장소의 isaaclab/을 중심으로 두고, 필요한 common/ 계약·tests/ 회귀 테스트·reports/windows/ 기록은 AGENTS.md 소유 경계를 지켜 변경한다. feat/isaaclab-windows 브랜치를 사용한다.

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

## 기존 Windows 인계의 첫 번째 종료점 — 완료 이력

아래는 기존 실행 경로를 열 때 사용한 관문이며 이미 완료한 이력입니다. 다음 개선 실험의 완료 조건이나 매번 처음부터 반복할 목록이 아닙니다.

1. 두 공통 검증 명령 통과
2. Isaac Sim, Isaac Lab, RSL-RL live check
3. SO-101 1환경 reset/step
4. SO-101 64환경 10,000 physics-step smoke
5. 64환경 PPO 10-iteration smoke와 run manifest

이는 정책 학습 성공이 아니라 학습 경로가 열린 상태입니다. 현재는 revision 2 기준선까지 실행했으므로, 다음 작업은 위 재개 메모의 실패 계측부터 이어갑니다. `docs/PROJECT_PLAN.md`는 기존 계획의 배경으로 참고합니다.
