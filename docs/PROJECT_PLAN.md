# SO-101 병렬 강화학습 프로젝트 계획

## 2026-09-12 목표 변경

최종 목표는 **Pick & Place 전체**다. 아래 LiftCube 단계는 기존 기반 검증 이력으로 유지하며,
현재 구현·GPU 실행 순서는 [Pick & Place 실행 계획](PICK_PLACE_PLAN.md)을 따른다.
신규 태스크 `SO101-PickPlace-v0`의 성공은 파지·들기·운반·목표 배치·그리퍼 해제·안정 유지까지다.
LiftCube G0~G4 PASS를 신규 태스크의 PASS로 재사용하지 않는다.

## 목표와 완료선

Windows의 Isaac Lab에서 SO-101 큐브 파지 PPO를 학습하고 Ubuntu의 MuJoCo에서 같은 정책을 교차평가합니다. 모델 하나를 얻는 것보다 학습이 성립한 조건과 실패한 조건을 다시 실행할 수 있게 만드는 것이 완료 기준입니다.

완료 결과에는 다음 네 가지가 있어야 합니다.

1. 환경 수에 따른 VRAM, steps/s, wall-clock 비교
2. PPO 3개 seed의 ID 및 held-out 성공률
3. reward ablation과 Domain Randomization 대조 결과
4. Isaac에서 학습한 checkpoint의 MuJoCo 평가 결과

실물 파지, 카메라 정책, BC와 ACT는 이 계획의 완료 조건이 아닙니다.

## PC별 역할

| 위치 | 작업 | 주요 산출물 |
| --- | --- | --- |
| Windows 11, RTX 3060 12GB | Isaac 환경, 보상, RSL-RL PPO, GPU profiling | checkpoint, train metrics, Isaac 평가 |
| Ubuntu | MuJoCo 환경, 공통 평가기, 결과 집계 | cross-sim 평가, 비교표, 그래프 |
| 공통 Git 저장소 | task contract, 설정, 작은 결과 파일 | commit과 artifact를 잇는 manifest |

Windows와 Ubuntu는 같은 저장소를 clone하지만 같은 파일을 동시에 편집하지 않습니다. Windows는 `isaaclab/`, Ubuntu는 `mujoco/`와 `evaluation/`을 맡습니다.

## 단계와 통과 조건

### G0. 저장소 계약

- `python scripts/validate_contract.py` 통과
- Windows와 Ubuntu의 contract SHA-256 일치
- 관절 순서, 단위, 성공 조건 고정

### G1. Windows 학습 스택 재검증

- Isaac Sim 4.5.0 Python smoke 통과
- Isaac Lab v2.1.1 commit 확인
- RSL-RL 2.3.3 CUDA import 확인
- 공식 Franka Lift random-agent가 1환경에서 reset과 step 수행

### G2. SO-101 1환경

- SO-101 USD와 큐브 장면 로드
- 관절 이름과 limit readback 저장
- zero action에서 비정상 가속, 관통, NaN 없음
- random action에서 각 환경의 큐브와 관절 상태가 독립적으로 변함

### G3. 64환경 smoke

- 64환경에서 10,000 physics step 완주
- reset 100회 동안 NaN과 invalid contact 없음
- 성공 판정은 강제 attachment 없이 접촉과 큐브 높이로 계산

### G4. PPO smoke

- 64환경, 10 iteration을 scratch로 실행
- checkpoint와 TensorBoard log 생성
- loss, reward, action std가 유한값
- 학습 성공을 주장하지 않고 학습 경로만 확인

### G5. 환경 수 사다리

- 64, 256, 512, 1024환경을 같은 짧은 예산으로 실행
- peak VRAM, collection steps/s, total steps/s 기록
- 2048환경은 1024보다 처리량이 좋아지고 메모리 여유가 있을 때만 실행
- 본 학습 환경 수는 처리량이 가장 높은 안정 지점으로 선택

### G6. PPO 기준선과 ablation

- 고정한 본 학습 예산으로 seed 0, 1, 2 실행
- random policy와 PPO를 같은 평가 프로토콜로 비교
- 보상 항 하나만 바꾸는 ablation 실행
- 실패 seed와 조기 종료도 결과에 포함

### G7. Domain Randomization

- DR off와 on을 같은 seed와 예산으로 실행
- 학습 중 범위와 held-out 범위를 분리
- 개선이 없으면 `no demonstrated improvement`로 기록

### G8. MuJoCo 교차평가

- Isaac과 동일한 관절 순서, action scaling, 성공 판정 적용
- 정책 입력의 의미와 정규화가 일치함을 fixture로 검증
- seed별 200에피소드 평가
- Isaac 대비 성공률 변화와 실패 단계를 보고

### G9. 포트폴리오 패키지

- 문제, 변경, 검증, 실패, 한계 순서의 결과 문서
- 환경 수 그래프, 학습 곡선, ablation 표, cross-sim 표
- 성공과 실패 영상 각각 최소 1개
- 모든 수치에 run ID와 manifest 연결

## 예상 작업량

| 구간 | 예상 범위 | 설명 |
| --- | --- | --- |
| G0-G1 | 반나절 | 기존 Windows 스택을 다시 확인 |
| G2-G3 | 1-3일 | SO-101 자산, 접촉, reset 검증 |
| G4-G6 | 3-7일 | 보상 수정과 PPO 기준선 |
| G7 | 1-3일 | DR 범위와 대조 실험 |
| G8 | 2-5일 | MuJoCo action/observation 정합과 평가 |
| G9 | 1-2일 | 결과 검증과 공개 자료 정리 |

기간은 설치 시간이 아니라 접촉과 보상 디버깅에 좌우됩니다. 각 구간은 앞 관문이 통과된 뒤 시작합니다.

## 중단 조건

- 64환경에서 non-finite state가 반복되면 PPO를 시작하지 않습니다.
- 성공 판정이 접촉 없이 참이 되면 보상 실험을 중단합니다.
- 환경 수를 늘렸는데 total steps/s가 떨어지면 이전 규모로 돌아갑니다.
- checkpoint 입력 형식이 contract와 다르면 MuJoCo 평가를 중단하고 변환기를 수정합니다.
- 실제 측정 없이 성공률이나 학습 시간을 README에 적지 않습니다.
