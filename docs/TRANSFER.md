# 저장소를 Windows로 전달하는 방법

## 현재 상태

공개 GitHub 저장소와 로컬 `origin`이 연결되어 있습니다.

- GitHub: https://github.com/mmporong/so101-pick-rl
- 기본 브랜치: `main`
- SSH remote: `git@github.com:mmporong/so101-pick-rl.git`

일상적인 동기화는 GitHub를 사용합니다. Git bundle은 네트워크 없이 전달하거나 복구할 때 쓰는 보조 수단입니다.

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

## GitHub에서 Windows로 전달

Windows에서는 공개 원격 저장소를 복제하고 작업 브랜치를 만듭니다.

```powershell
git clone https://github.com/mmporong/so101-pick-rl.git "$HOME\so101-pick-rl"
Set-Location "$HOME\so101-pick-rl"
git switch -c feat/isaaclab-windows
python scripts\validate_contract.py
python -m unittest discover -s tests -v
```

Ubuntu는 `feat/mujoco-linux` 브랜치를 사용합니다. 공통 계약은 한쪽에서만 변경한 뒤 main에 반영합니다.

각 PC에서 작업을 시작하기 전에는 다음 명령으로 최신 `main`을 확인합니다.

```bash
git fetch origin
git status --short --branch
```

## Git으로 보내지 않는 파일

- checkpoint
- HDF5 dataset
- TensorBoard 원본 로그
- Isaac Sim cache
- MP4 원본

이 파일은 Hugging Face private 저장소나 별도 artifact 경로에 두고, Git에는 URI와 SHA-256을 기록합니다.
