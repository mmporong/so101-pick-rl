# Pick & Place 구현·실행 증거

2026-09-12부터의 기록이다. 기존 `STATUS.md`는 LiftCube 이력으로 보존한다. 신규 `SO101-PickPlace-v0`의 전체 성공 조건은 `common/pick_place_spec.json` revision 2다. 실물 실행은 하지 않았다. 이후 사용자의 push 요청에 따라 코드와 작은 검증 기록만 원격에 반영했으며 MP4·학습 원본 데이터는 로컬에 남겼다.

## 구현 검증

2026-09-13 파지 진단 추가: 사용자 지적에 따라 최신 model_389를 고정해 PhysX 접촉점·법선력·
분리거리를 읽었다. 첫 seed에서 힘을 받는 gripper–cube 접촉에 약 -25.4mm의 분리거리가
기록됐다. 이전 '유효한 들기' 집계는 코드의 접촉·높이·유지 조건 통과이지 정상 파지의
증거가 아니다. [접촉 진단 보고서](pp_grasp_contact_audit_20260913.json)를 우선 참고한다.
계측 ON/OFF 90전이의 관절·물체·명령·접촉·상태는 정확히 일치했다. 정책·보상·물체 크기·
시뮬레이터 설정은 변경하지 않았고 재학습하지 않았다. 진단 코드 포함 CPU 테스트는 82개 PASS다.

2026-09-13 최신: 약 37° 사선 전체 화면 → 10초 줌인 → 16환경 구역 근접 선회로 재촬영했다.
clean source `9d300327`에서 실제 GPU PPO 1,024환경·50 update(iteration 340–389)를 수행했다.
1920×1080·30fps·40초 MP4와 별도 model_389를 저장하고, source model_339는 보존했다.
선택 당시 16환경 중 15환경의 유효한 들기 이력과 전체 성공 0을 관찰했다. 선정 편향이 있는
촬영 기록이지 성공률 평가가 아니다. [시네마틱 보고서](pp_cinematic_video_20260913.json) 참조.
CPU 75개 PASS, 전체 영상 디코드·해시·프레임·최종 구도 검증 PASS. MP4는 로컬에만 보관한다.

2026-09-13 이전 촬영: 실제 PPO 1,024환경·40 update 재개 촬영을 완료했다.
clean source `e40bc5c`에서 1280×720·30fps·32초 MP4를 생성하고 원본 모델을 보존했다.
[촬영 보고서](pp_parallel_video_20260913.json)에 실행·파일 해시를 기록했다.
MP4는 로컬 Git 비추적 자료이며 업로드하지 않는다. 당시 CPU 테스트는 71개 PASS다.
정책 개선 및 새 모델의 정식 성공률 평가는 이번 촬영 범위에 포함하지 않았다.

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
- 당시 `--capture --enable_cameras` 진단은 Isaac viewport 초기화의 access violation으로 JSON 생성 전에 종료됐다. 배치 exit code만으로 성공을 판단하지 않았다. 2026-09-13에는 설치본 수정 없이 D3D12 경로로 실제 영상 촬영을 완료했다(위 촬영 보고서).
- `pp_full_fixture_diagnostic_20260912.json` 및 이후 review fixture는 합성 과거 이력을 주입해 실제 안정 배치·최종 종료·자동 reset 연결을 검사한다. 정책이 직접 집고 놓았다는 증거가 아니다.

## 현재 남은 작업

실행 기반 검증과 revision 2 원본 model_299의 seed 0/1/2 각 200회 ID 평가는 완료했다. 해당 모델의 전체 Pick & Place 성공은 0/600으로 미달이다. 남은 핵심은 들어 올린 물체를 목표 높이로 내려놓고 해제하는 학습이다. DR/ablation/held-out/MuJoCo 교차평가는 미실행이다. 병렬 학습 영상은 검증했지만 촬영 후 model_339와 model_389의 정식 성공률 평가는 미실행이다.

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

## revision 2 최종 baseline

- source commit: `a678ceda7dad1b9bc6d1c50c28a663fcc83fdb21`
- contract SHA: `f4af446ce9644e558653da595d34b0c752dde974e3951051dee7d8d672dc52d3`
- G2 1환경/1,000 physics step, G3 64환경/10,000 physics step, 물리 fixture, G4 64환경/10 iteration, checkpoint restart 모두 새 commit의 clean 상태에서 PASS.
- 1,024환경 scratch PPO 300 iteration: 7,372,800 transitions, 학습 716.6068초, 최대 VRAM 3,557MiB, 평균 GPU 41.38%, 최고 GPU 53%. 환경과 정책은 모두 cuda:0.
- 관측/보상 비유한 값과 non_finite 종료 0. 학습 중 작업 영역 이탈 2,389회는 실패 에피소드로 기록했다. 최종 성공 종료 0회.
- checkpoint: `isaaclab/logs/rsl_rl/so101_pick_place/2026-09-12_13-48-21_pp_r2_baseline300/model_299.pt`
- checkpoint SHA: `d8d87329b4eeb3f2b45997aa3b61fc5eff6e411e9d2500b2711927ab4d26b35e`
- checkpoint·TensorBoard·저장된 agent.yaml·run_contract.json은 로컬에 보존한다. 대용량 파일은 Git에 넣지 않았다.

### 독립 ID 평가: 동일 checkpoint, seed별 200회

학습 프로세스를 종료한 뒤 별도 평가기가 checkpoint를 다시 읽었다. 각 seed에서 256환경을 생성하고 고정 quota로 정확히 200회만 집계했다. non_finite와 workspace_exit도 실패 분모에 포함했다. 각 원본 JSON은 단일 seed diagnostic이라고 표시하며, 아래 3개 결과를 모두 확인해 계약의 ID 평가 세트를 완성했다. 이는 정책 성능 목표 달성을 뜻하지 않는다.

| seed | 평가 수 | 전체 성공 | 접촉했으나 들기 미달 | 든 뒤 목표 배치 미달 | 영역 이탈 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 0 | 27 | 172 | 1 |
| 1 | 200 | 0 | 20 | 179 | 1 |
| 2 | 200 | 0 | 24 | 174 | 2 |
| 합계 | 600 | 0 | 71 | 525 | 4 |

전체 성공률은 **0/600 = 0%**다. 525회(87.5%)는 양손 접촉으로 8cm 들기·0.2초 유지 이력까지 확인됐지만 최종 XYZ 배치 허용오차에 도달하지 못했다. 영역 이탈 4회는 우선 실패 분류가 적용되므로 이 표만으로 그 4회의 이전 파지 여부를 추정하지 않는다. 따라서 87.5%를 별도로 완결된 Pick 태스크 성공률이라고 과장하지 않는다.

원본은 `pp_r2_eval_seed0.json`~`pp_r2_eval_seed2.json`, 공통 evaluation_result schema 투영본은 `pp_r2_id_seed0.json`~`pp_r2_id_seed2.json`이다. `status=passed/completed`는 실행·집계의 정상 완료이며, `successes=0`인 정책을 성공했다고 표시한 것이 아니다.

현재 증거로는 **집기 이후 목표 배치/해제가 병목**임을 확인할 수 있다. 정확한 원인은 추가 trajectory 진단 없이는 단정하지 않는다. 기존 접근/정렬 보상 좌표와 실제 그리퍼 중앙의 관계, 목표 위 하강과 해제 학습은 후속 검증 대상으로 남긴다. 성공 기준을 낮추거나 기존 평가를 제외하지 않았다.
