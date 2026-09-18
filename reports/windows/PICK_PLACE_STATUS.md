# Pick & Place 구현·실행 증거

2026-09-12부터의 기록이다. 기존 `STATUS.md`는 LiftCube 이력으로 보존한다. 현재 `SO101-PickPlace-v0` 계약은 `common/pick_place_spec.json` revision 3다. 실물 실행은 하지 않았다. MP4·학습 원본 데이터는 로컬에 남긴다.

## 2026-09-18 정지 파지·들기 비교

정상 들기는 아직 실패이며 revision 3 PPO는 `not_run`이다. 보상·물리·성공 조건을 바꾸지 않고 진단에 정지 유지, 느린 수직 목표, 접근 높이 비교, 들기 시작 자세 기준 옵션을 추가했다. 미세 닫힘이 일찍 시작돼도 전체 관절 범위를 이동할 수 있도록 닫기 스텝 예산을 계산한다. 측정되지 않은 팁 높이는 자세 유지 기준으로 허용하지 않는다.

커밋 `80f5bdaf4b599e32cc3804702a4730ae53a4b5d1`의 clean 상태에서 seed 0·GPU 1환경으로 비교했다. 두 실행은 초기 큐브 위치와 들기 전 단계 결과가 동일하다.

| 진단 | 책상 위 정지 유지 | 들기 결과 | 전체 성공 |
| --- | --- | --- | --- |
| 2초 유지 후 기존 step 목표 | 60스텝 통과 | 4스텝째 양측 접촉 상실 | 실패 |
| 2초 유지 후 0.01m/s 목표 | 60스텝 통과 | 7스텝째 양측 접촉 상실 | 실패 |

정지 중 6개 관절 목표는 고정됐지만 큐브는 60스텝 모두 책상에 닿아 있었다. 공중 파지 유지의 증거가 아니다. 느린 목표는 큰 횡방향 이탈을 줄였으나 큐브 상승은 성립하지 않았다. 두 실행의 최대 finger–cube 음수 분리거리는 0.346mm로 1mm 진단 한계 이내이며 finger–table 접촉은 관측되지 않았다.

들기 직전 힘을 받는 접촉점은 고정 손가락이 높이 18.18mm, 이동 손가락이 39.96mm였다. 큐브 윗면은 약 40mm이며 이동 손가락 접촉 법선의 Z 성분은 0.495였다. 양쪽 옆면을 마주 잡기보다 한쪽이 윗모서리에 걸리는 비대칭 접촉을 우선 의심한다. 이는 cooked 충돌 형상의 오류나 유일 원인을 확정한 결과는 아니다.

개발 중 추가 비교에서도 접근 위치를 8mm 낮추면 정지 유지 뒤 들기 2스텝째, 들기 시작의 자세 기준·팁 높이 차이를 유지하면 6스텝째 접촉을 잃었다. 후자는 전체 손목 방향 고정이나 자세 보정 gain 0 실험이 아니다. 다음 순서는 실제 안쪽 파지 면의 좌표·접촉점·대향 법선 확인 → 정상 공중 유지와 전체 집기/놓기 검증 → revision 3 scratch 학습이다. 근거 없이 힘·마찰·침투 한계를 높이지 않는다.

검증: 계약·CPU 136개·컴파일·diff 검사 PASS. 독립 code-reviewer `COMMENT`(비차단), architect `CLEAR`; 신규 미해결 결함은 없다. 검토 Python 파일의 기존 외부 import·동적 config 관련 Pyright 오류 5건은 남아 전체 정적 분석 PASS로 표시하지 않는다. 환경·보상은 바꾸지 않아 64환경 smoke는 이번에 재실행하지 않았다. 원본 JSON/trace는 로컬 `artifacts/`에 보존하고 [실행 경로·SHA 및 비교 보고서](pp_retention_diagnostics_20260918.json)에 연결했다. 영상·체크포인트·원본 trace는 Git에 넣지 않는다.

## 2026-09-13 revision 3 개선

- 접근·정렬·접촉·들기·이동·내려놓기·열기·후퇴·안정화의 9개 보상 항목을 분리했다. 하나의 PPO 정책이 가중 합을 받으며, 단계마다 별도 정책을 학습하는 구조가 아니다.
- 각 항목은 절대 상태 점수가 아니라 `gamma * 다음 potential - 이전 potential`이다. 완료한 단계의 potential은 유지하고, 후퇴에는 음수 차이를 적용한다. 제자리 유지·단계 왕복으로 양의 보상을 반복 수집하지 못하도록 했다. 진짜 종료에서는 다음 potential을 0으로, timeout에서는 bootstrap을 위해 유지한다.
- 목표 위에서 접촉을 유지한 채 그리퍼를 여는 전이에 보상을 추가했다. 최종 성공 보상 1,000과 기존 높이·속도·안정 시간 조건은 낮추지 않았다.
- 큐브 근처에서 무접촉 개방을 먼저 관찰한 뒤 양손가락 접촉과 닫힘을 만족해야 들기 이력을 인정한다. 개방/파지 이력 두 항목을 관측에 추가해 39→41차원으로 변경했다. 이 순서는 필요조건이며 실제 끼움·충돌 형상이 정상이라는 충분조건이 아니다.
- 계약 SHA는 `fe1ad4d8d9df88a63649a8d9183d91c1b11f5c460cb6bd7bfacfc84583d4b929`다. 기존 39차원 체크포인트는 그대로 보존하며 새 계약으로 자동 resume하지 않는다. LiftCube 계약과 4cm 큐브는 변경하지 않았다.
- `probe_grasp_feasibility.py`는 물체를 붙이거나 순간 이동시키지 않는 스크립트형 물리 진단이다. 기본 파지 프레임과 출처 해시가 있는 손끝 중간점/높이 정렬 진단을 비교한다. 학습 정책 성공률 측정은 아니다.
- PickPlace 학습에는 `--grasp_feasibility_report`가 필요하다. 동일한 clean source commit·계약, 모든 파지/배치 게이트, 원래 충돌 설정을 확인한다. 진단 한계를 완화한 보고서는 학습을 승인하지 못한다. 보고서는 `artifacts/` 같은 Git 비추적 경로에 생성한다.
- 기본 침투 한계 1mm, 닫기 전 이동 12mm, 손목 굽힘 75°는 보수적인 진단 설계값이지 실물 캘리브레이션 수치가 아니다. 접촉 기록은 30Hz의 마지막 물리 substep 표본이므로 모든 120Hz 접촉이나 cooked convex hull의 정합성을 증명하지 않는다.

정상 물리 파지가 확인되기 전에는 revision 3 PPO를 실행하지 않는다. 단위 테스트·smoke·합성 성공 fixture의 PASS를 정상 파지 또는 정책 성공으로 해석하지 않는다.

검증: 계약·CPU 127개·컴파일 PASS, GPU 1환경 1,000스텝 및 64환경 10,000스텝 PASS(자동 reset 256회, 실패 0), 합성 fixture 18항목 PASS. 실행은 개발 중 dirty source의 진단이며 clean-source 정책 성능 기록이 아니다. 원본 경로·SHA와 실패 비교는 [revision 3 검증 보고서](pp_r3_staged_rewards_20260913.json)에 남긴다.

최종 물리 진단은 실패다. 손끝 중점/높이 정렬과 접촉 후 목표 고정을 적용한 probe04에서는 finger–cube 최대 음수 분리거리가 0.346mm로 진단 한계 1mm 이내였고 finger–table 접촉은 관측되지 않았다. 그러나 lift 6번째 정책 스텝에 양측 접촉을 잃어 즉시 중단했다. 큐브는 들리지 않았고 운반·해제·전체 성공도 성립하지 않았다. 이전 probe03의 10.31mm 깊은 압착과 빈손 후속 동작은 이 진단에서 방지했지만, 학습 정책 전반의 침투 문제 해결을 뜻하지 않는다. 다음 물리 검증은 파지 유지력·접촉 방향·들기 경로와 파지 프레임이며, 근거 없이 마찰·힘·충돌 허용치를 높이지 않는다. revision 3 학습 및 정책 평가는 `not_run`이다.

독립 검토는 code-reviewer `COMMENT`(비차단), architect `CLEAR`, 종합 `COMMENT`다. 새 미해결 결함은 없지만 Pyright 1.1.414 전체 변경 Python 검사에는 Isaac 동적 config·런타임 import 관련 진단 73개가 남아 전체 정적 분석 PASS로 주장하지 않는다. 순수 로컬 모듈·테스트 묶음은 0 errors다.

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

revision 2 원본 model_299의 seed 0/1/2 각 200회 ID 평가는 전체 Pick & Place 0/600이다. 최신 접촉 진단에 따라 우선순위는 정상 파지 중심·접근 자세와 물리 접촉 검증 → revision 3 scratch 학습 → 고정 seed 전체 성공 평가 순서다. 보상 변경만으로 깊은 겹침 원인이 해결됐다고 주장하지 않는다. DR/ablation/held-out/MuJoCo 교차평가와 model_339/model_389 정식 성공률 평가는 미실행이다.

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
