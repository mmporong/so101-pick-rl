# 저장소를 Windows로 전달하는 방법

## 현재 상태

이 저장소는 로컬 Git 저장소입니다. 외부 GitHub 저장소와 remote는 아직 만들지 않았습니다.

장기 작업은 private GitHub remote를 연결하는 방식이 편합니다. 외부 remote를 만들기 전에도 Git bundle 파일 하나로 Windows에 전달할 수 있습니다.

## Git bundle로 전달

Ubuntu에서 만든 `so101-pick-rl-main.bundle`을 USB, 개인 클라우드 또는 두 PC 사이의 파일 전송 수단으로 Windows에 복사합니다.

Windows PowerShell에서 bundle이 있는 폴더로 이동한 뒤 실행합니다.

```powershell
git clone .\so101-pick-rl-main.bundle "$HOME\so101-pick-rl"
Set-Location "$HOME\so101-pick-rl"
git status --short --branch
python scripts\validate_contract.py
python -m unittest discover -s tests -v
```

그다음 `HANDOFF.md`와 `docs/HANDOFF_WINDOWS.md`를 순서대로 읽습니다.

bundle은 생성 시점의 commit까지만 포함합니다. Ubuntu에서 새 commit을 만든 뒤에는 bundle도 다시 생성해야 합니다.

## private GitHub remote를 연결한 뒤

외부 저장소 생성과 push가 허용된 세션에서 remote를 연결합니다.

```bash
cd "$HOME/so101-pick-rl"
git remote add origin <PRIVATE_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Windows에서는 remote에서 clone하고 작업 브랜치를 만듭니다.

```powershell
git clone <PRIVATE_GITHUB_REPOSITORY_URL> "$HOME\so101-pick-rl"
Set-Location "$HOME\so101-pick-rl"
git switch -c feat/isaaclab-windows
```

Ubuntu는 `feat/mujoco-linux` 브랜치를 사용합니다. 공통 계약은 한쪽에서만 변경한 뒤 main에 반영합니다.

## Git으로 보내지 않는 파일

- checkpoint
- HDF5 dataset
- TensorBoard 원본 로그
- Isaac Sim cache
- MP4 원본

이 파일은 Hugging Face private 저장소나 별도 artifact 경로에 두고, Git에는 URI와 SHA-256을 기록합니다.
