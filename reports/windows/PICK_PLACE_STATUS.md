# Pick & Place 구현·실행 증거

2026-09-12. 기존 `STATUS.md`는 LiftCube 이력으로 보존한다. 신규 `SO101-PickPlace-v0`의 전체 성공 조건은 `common/pick_place_spec.json`이다. 실물 실행과 원격 push는 하지 않는다.

## 구현 검증

- 계약 검증 PASS. 기존 LiftCube SHA는 변경하지 않았다.
- CPU 단위 테스트 57개 PASS. Isaac Sim bundled Python의 `SO101_TEST_DEVICE=cuda` 상태·계약 테스트 32개 PASS. 부분 reset 단독 테스트는 CPU에서도 별도로 수행하고 실제 CUDA 환경 fixture로 격리를 검증한다.
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
