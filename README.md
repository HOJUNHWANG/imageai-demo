# ImageAI Demo (SDXL Inpaint)

AI 기반 이미지 편집 데모 앱입니다. **마스크 기반(inpainting)** 워크플로우를 중심으로,
SDXL + (선택) ControlNet + (선택) SAM/MediaPipe 자동 마스크를 결합했습니다.

- UI: Gradio (로컬 실행)
- 모델/가중치 파일은 **레포에 포함되어 있지 않습니다** (직접 다운로드 필요)
- (선택) 데모 이미지/스크린샷은 `assets/` 폴더에 추가하는 것을 권장합니다.

## 주요 기능 (코드 기준)
- **Inpainting 편집**: 마스크 영역만 생성/수정
- **마스크 생성**
  - Manual: **SAM** 클릭 기반 마스크 (vit_b / vit_h)
  - Auto: **MediaPipe Tasks** 기반 semantic 마스크 (상의/소매/바지/머리/배경)
- **ControlNet (선택)**: depth / openpose 로컬 로딩 경로 지원
- **Refine pass (선택)**: img2img 기반 톤/디테일 정리
- **Prompt enrich**: positive/negative 자동 확장 + Preview 기능
- **Seed 제어**: 고정/랜덤
- **VRAM 가드레일**: public demo 모드에서 long side / steps 제한

> 참고: README에는 “public demo에서 특정 기능 비활성화”를 언급하지만,
> 현재 코드상으로는 주로 **리소스(해상도/스텝/큐) 제한 형태**로 구현되어 있습니다.

## 요구사항
- Python 3.11
- (권장) NVIDIA GPU + CUDA (예: RTX 30xx, VRAM 12GB+)
- CPU도 가능하지만 매우 느릴 수 있습니다.

## 설치 (Windows 기준)
```bash
python -m venv venv311
.\venv311\Scripts\activate

pip install -r requirements.txt
# SAM 설치 (권장: git 설치)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 모델/가중치 준비(직접 다운로드)
아래 파일/폴더는 예시 경로이며, **코드에서 참조하는 기본 경로**는 다음과 같습니다.

- SDXL inpaint 베이스(기본): `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- (옵션) Juggernaut XL safetensors 경로(코드 기준):
  - `models/stable-diffusion-xl/juggernautXL_ragnarokBy.safetensors`
- (옵션) ControlNet 로컬 폴더:
  - `models/ControlNet/controlnet-depth-sdxl-1.0`
  - `models/ControlNet/controlnet-openpose-sdxl-1.0`
- (옵션) SAM weights:
  - `weights/sam_vit_b_01ec64.pth`
  - `weights/sam_vit_h_4b8939.pth`
- MediaPipe: 첫 실행 시 자동 다운로드

## 실행
```bash
python app.py
```
브라우저에서 열기: http://127.0.0.1:7860

## Public demo 설정 (환경변수)
코드에 실제로 존재하는 플래그 기준:
- `PUBLIC_DEMO=1` (기본값): 공개 데모 가드레일 ON
- `PUBLIC_MAX_LONG_SIDE=896` (기본값)
- `PUBLIC_MAX_STEPS=22` (기본값)
- `PUBLIC_MAX_QUEUE=10` (기본값)
- `PUBLIC_CONCURRENCY=1` (기본값)
- `MOCK_INPAINT=1`: 모델 실행 없이 UI/파이프라인만 빠르게 테스트

`.env.example`을 참고해 `.env`를 만들 수 있습니다. **토큰/키는 절대 커밋하지 마세요.**

## 알려진 이슈/주의
- CPU 모드: ControlNet/SAM 사용 시 RAM 사용량이 커질 수 있습니다.
- Auto mask(MediaPipe)는 SAM 대비 정밀도가 낮을 수 있습니다.

## License
MIT License (see `LICENSE`).
