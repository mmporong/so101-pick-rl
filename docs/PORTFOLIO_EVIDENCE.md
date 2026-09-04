# 포트폴리오 증거 장부

## 현재 공개 가능한 상태

| 주장 | 상태 | 근거 |
| --- | --- | --- |
| Windows와 Ubuntu가 공유하는 task contract를 만들었다 | implemented | `common/task_spec.json`, validator tests |
| SO-101을 Isaac Lab에 등록했다 | not_run | G2 결과 필요 |
| PPO로 큐브 파지를 학습했다 | not_run | G6 checkpoint와 평가 필요 |
| RTX 3060에서 병렬 처리량을 측정했다 | not_run | G5 scale report 필요 |
| reward 효과를 분리했다 | not_run | G6 ablation report 필요 |
| Domain Randomization을 검증했다 | not_run | G7 대조 결과 필요 |
| MuJoCo 교차평가를 완료했다 | not_run | G8 결과 필요 |
| 실물 SO-101에 정책을 배포했다 | out_of_scope | 별도 실물 프로젝트 범위 |

현재 README에는 설계와 범위만 적습니다. `not_run` 항목은 측정 결과 영역이나 이력서 성과 문장에 사용하지 않습니다.

## 실행별로 남길 자료

```text
reports/
├── windows/
│   ├── w0_environment.json
│   ├── g3_physics_smoke.json
│   ├── g5_scale_ladder.csv
│   ├── g6_ppo_summary.json
│   └── g7_dr_ablation.csv
├── linux/
│   ├── contract_fixture.json
│   └── g8_cross_sim.csv
└── portfolio/
    ├── results_summary.md
    ├── failure_gallery.md
    └── media_manifest.json
```

파일은 실제 실행이 끝날 때 생성합니다. 비어 있는 결과 파일을 미리 만들지 않습니다.

## 결과 문서의 순서

포트폴리오 본문은 다음 순서로 작성합니다.

1. 막힌 문제: SO-101의 작은 그리퍼와 접촉을 포함한 PPO가 안정적으로 학습되는가?
2. 바꾼 것: 관측, action, reward, reset, 환경 수 중 실제 수정한 항목
3. 확인한 결과: run ID가 연결된 성공률과 처리량
4. 남은 실패: 실패 단계와 재현 조건
5. 현재 한계: privileged state, simulation-only, 실물 미검증
6. 다음 실험: perception 입력 또는 실물 system identification

## 필요한 시각 자료

- 1환경 contact debug 영상
- 512개 이상 병렬환경 화면
- reward와 success-rate 학습 곡선
- 환경 수별 steps/s 및 VRAM 그래프
- reward ablation 표
- Isaac-MuJoCo 성공률과 실패 단계 비교
- 성공 영상 1개와 대표 실패 영상 1개

병렬환경 화면은 규모를 보여 주는 보조 자료입니다. 성능 근거는 CSV와 run manifest에서 가져옵니다.

## 완료 후 사용할 수 있는 문장 구조

수치는 실제 결과 파일에서 채웁니다.

```text
SO-101 큐브 파지 태스크를 Isaac Lab 외부 extension으로 구현하고 RSL-RL PPO를 학습했다.
환경 수별 GPU 처리량을 측정해 본 학습 규모를 선택했으며, reward ablation으로 정책 행동을 바꾼 항목을 분리했다.
같은 checkpoint를 MuJoCo에서 평가해 simulator 변경에 따른 성공률과 실패 단계 변화를 기록했다.
```

실물 평가가 없다면 마지막 문장에 simulation-only 한계를 함께 표시합니다.
