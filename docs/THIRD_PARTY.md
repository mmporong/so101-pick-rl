# 외부 자산과 코드 출처

## 사용 예정 자료

| 자료 | 출처 | 고정 버전 | 현재 상태 | 사용 범위 |
| --- | --- | --- | --- | --- |
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | v2.1.1 | Windows 외부 설치 확인 대상 | 환경 API와 RSL-RL wrapper |
| RSL-RL | https://github.com/leggedrobotics/rsl_rl | 2.3.3 | Windows 설치 확인 대상 | PPO |
| LeIsaac | https://github.com/LightwheelAI/leisaac | v0.1.2, commit `47df88fee80b1d8b27faf1632ee500c63c4cf1c7` | reference_only | SO-101 설정과 LiftCube 환경 참고 |
| ROBOTIS cyclo_lab | https://github.com/ROBOTIS-GIT/cyclo_lab | main 조사 시점 | reference_only | RL과 IL 공개 워크플로 비교 |
| 기존 SO-101 MuJoCo 미러 | `$HOME/so101-mobile-manipulation/sim` | 가져오는 시점의 Git commit 기록 필요 | reference_only | MuJoCo 모델과 좌표계 참고 |

`reference_only`는 아직 파일을 복사하거나 설치 결과를 이 저장소의 구현으로 사용하지 않았다는 뜻입니다.

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
