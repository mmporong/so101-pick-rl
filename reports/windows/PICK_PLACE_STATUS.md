# Pick & Place 구현·실행 증거

2026-09-12. 기존 `STATUS.md`는 LiftCube 이력으로 보존한다. 신규 `SO101-PickPlace-v0`의 전체 성공 조건은 `common/pick_place_spec.json` revision 2다. 실물 실행과 원격 push는 하지 않는다.

## 구현 검증

- 계약 검증 PASS. 기존 LiftCube SHA는 변경하지 않았다.
- revision 2 CPU 단위 테스트 58개 PASS. Isaac Sim bundled Python의 `SO101_TEST_DEVICE=cuda` 상태·계약 테스트 33개 PASS. 부분 reset 단독 테스트는 CPU에서도 별도로 수행하고 실제 CUDA 환경 fixture로 격리를 검증한다.
- 목표 밖 한 손가락 밀기, 고속·고각속도 해제, 목표 높이 오류, 샘플 사이 던지기/정착, 해제 후 튀기기, 파지 이력 없는 배치를 성공으로 세지 않는다.
- 전체 성공의 1회 보상과 dense reward/할인율은 계약 SHA에 포함한다. 학습 성공률을 측정한 수치가 아니다.

## 초기 진단 기록의 해석

`pp_*diagnostic*_20260912.json` 및 `pp_geometry_*_20260912.json`은 미커밋 구현을 점검한 개발 기록이다. `git_dirty=true`이며 최종 clean-source 실행이나 정책 성능의 증거로 사용하지 않는다. 실패·불충분한 시도를 삭제하거나 성공으로 바꾸지 않았다.

- 첫 `pp_fixture_diagnostic_20260912.json`의 개방 비교는 회전축 근처 body origin을 써 불충분했다.
- `pp_fixture_diagnostic2_20260912.json`, `pp_geometry_proxies_20260912.json`의 임의 probe는 실제 손가락 끝을 대표하지 못했다.
- `pp_geometry_headless_20260912.json`에서는 instance proxy 메시를 아직 읽지 못했다.
- `pp_geometry_mesh_20260912.json`부터 실제 메시 끝부분의 body-local 좌표를 사용한다. 닫힘/열림 끝부분 간격은 약 0.01049m/0.13726m로 측정됐다.
- `--capture --enable_cameras` 진단은 Isaac viewport 초기화의 access violation으로 JSON 생성 전에 종료됐다. 배치 exit code만으로 성공을 판단하지 않았다. 화면 없는 물리 경로는 정상 동작했으며, 영상 증거는 아직 없다.
- `pp_full_fixture_diagnostic_20260912.json` 및 이후 review fixture는 합성 과거 이력을 주입해 실제 안정 배치·최종 종료·자동 reset 연결을 검사한다. 정책이 직접 집고 놓았다는 증거가 아니다.

## 남은 실제 실행

clean source commit의 1환경/64환경 smoke, PPO 학습·재시작, 환경 수별 자원 비교, 저장된 checkpoint의 독립 평가를 순서대로 진행한다. 전체 정책 성공, seed 0/1/2 각 200회 정식 평가, DR/ablation/MuJoCo 교차평가는 아직 완료로 주장하지 않는다.

## revision 1 clean-source 실행 및 중단

source `ddb1583a4a6fd78a0e93865e0dcf7b238e914f09`, contract `5cef9bc9bd04fb98678b65758e043dbe419e6044992eb4e740765db2d9c46de1`에서 실행했다. `pp_r1_*.json`은 당시 보고서의 원본 사본이다.

- G2: 1환경 1,000 physics step PASS.
- G3: 64환경 10,000 physics step PASS, 자동 reset 256회, reset 실패/비유한/작업영역 이탈 0.
- G4: 64환경 10 iteration, 15,360 transitions, 21.4773초 PASS. 실제 환경·정책은 cuda:0.
- checkpoint restart: iteration 9에서 10/11로 재개 PASS.
- 독립 평가기 smoke: 16/16에피소드 집계 PASS, 정책 성공은 0/16이며 모두 reached_not_grasped.
- clean-source 물리 fixture PASS. 합성 이력 기반 검사이지 learned-policy 성공이 아니다.

| 환경 수 | 10회 학습 전이 수 | 학습 초 | 전이/초 | 최대 VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| 64 | 15,360 | 21.4773 | 715 | 3,151 |
| 256 | 61,440 | 21.4219 | 2,868 | 3,221 |
| 512 | 122,880 | 23.0451 | 5,332 | 3,370 |
| 1,024 | 245,760 | 29.1635 | 8,427 | 3,610 |

1,024환경을 선택해 300 iteration scratch 학습을 시작했으나 107회 로그(iteration 0~106) 뒤 Ctrl-C로 중단했다. 정상 하강에서 큐브 중심 높이 30~32mm 구간이 운반 이력을 잘못 지우는 문제가 CPU 재현으로 확인됐기 때문이다. iteration 100 checkpoint와 TensorBoard를 보존했고, 원본 final report는 Ctrl-C 시 생성되지 않아 별도 `pp_baseline300_r1_interrupted.json`에 재구성 근거와 누락을 명시했다. 이 실행은 성공/수렴 학습으로 계산하지 않는다.

revision 2는 XY 목표 영역 안의 정상 하강을 복구한다. 성공 XYZ 조건·해제 속도·안정 시간·목표 밖 끌기 금지는 유지한다. 계약 SHA가 바뀌었으므로 revision 1 checkpoint를 자동 재개하지 않는다.

독립 리뷰: code-reviewer COMMENT(미해결 blocker 0, LSP/pyright/ruff 부재), architect CLEAR. 따라서 시스템 baseline 진입은 가능하나 정식 정적 분석까지 완료했다고 주장하지 않는다.
