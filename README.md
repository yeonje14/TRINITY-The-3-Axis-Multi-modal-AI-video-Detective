# 🛡️ TRINITY v2: 3-Axis Ensemble AI-Generated Media Detector

> **"신호가 말하고, 출처가 증언하며, 의미가 검증한다."**
> **"Signals speak. Provenance testifies. Semantics verify."**
> **「信号が語り、出所が証言し、意味が検証する。」**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch + CoreML](https://img.shields.io/badge/PyTorch%20MPS%20%2B%20CoreML%20ANE-Apple%20Silicon-000000.svg)](https://pytorch.org/)
[![Server: FastAPI](https://img.shields.io/badge/Server-FastAPI%20%2B%20Celery-009688.svg)](https://fastapi.tiangolo.com/)
[![Client: Flutter](https://img.shields.io/badge/Client-Flutter%20(Android%2FiOS)-02569B.svg)](https://flutter.dev/)

---

### 🌍 Language Select
[🇰🇷 **한국어 (Korean)**](#-한국어-korean) | [🇺🇸 **English**](#-english) | [🇯🇵 **日本語 (Japanese)**](#-日本語-japanese)

---

> ⚠️ **v2 Redesign (2026-07) / Research Prototype**
> 카카오톡 챗봇 인터페이스를 폐기하고 **Flutter 앱 + REST API** 구조로 전환했으며, 탐지 엔진을 2024–2026 최신 문헌 기반의 **신호(Signal) · 출처(Provenance) · 의미(Semantic) 3축**으로 전면 재설계했습니다.
> The KakaoTalk chatbot interface has been retired in favor of a **Flutter app + REST API**, and the detection engine has been fully redesigned around three axes — **Signal · Provenance · Semantic** — grounded in 2024–2026 literature.
> カカオトークチャットボットを廃止し、**Flutterアプリ + REST API** 構成へ移行。検知エンジンは2024–2026年の最新文献に基づく**信号・出所・意味の3軸**として全面再設計されました。

---

## 📂 Project Structure

```
TRINITY/
├── 📂 interfaces/
│   └── 📂 api/                  # [KR] Flutter용 REST API / [EN] REST API for Flutter client / [JP] Flutter向けREST API
│       ├── routes.py            # [KR] 엔드포인트 / [EN] Endpoints / [JP] エンドポイント
│       └── schemas.py           # [KR] 요청·응답 스키마 / [EN] Request/response schemas / [JP] リクエスト・レスポンススキーマ
│
├── 📂 core/                     # [KR] 3축 탐지 엔진 / [EN] 3-Axis Detection Engine / [JP] 3軸検知エンジン
│   ├── 📂 axis_signal/          # [KR] 1축: 신호(주파수) / [EN] Axis 1: Signal (Frequency) / [JP] 第1軸: 信号(周波数)
│   │   ├── spectrum.py          # [KR] FFT/DCT 스펙트럼 잔차 / [EN] FFT/DCT spectral residual / [JP] FFT/DCTスペクトル残差
│   │   ├── hybrid_attention.py  # [KR] 공간-주파수 하이브리드(EfficientNet+CBAM) / [EN] Spatial-frequency hybrid / [JP] 空間-周波数ハイブリッド
│   │   └── freq_masking.py      # [KR] 주파수 마스킹 학습 / [EN] Frequency-domain masking / [JP] 周波数マスキング学習
│   ├── 📂 axis_provenance/      # [KR] 2축: 출처 증명 / [EN] Axis 2: Provenance / [JP] 第2軸: 出所証明
│   │   ├── c2pa.py              # [KR] C2PA 매니페스트 검증 / [EN] C2PA manifest verification / [JP] C2PAマニフェスト検証
│   │   └── synthid.py           # [KR] SynthID 플러그인 슬롯 / [EN] SynthID plugin slot / [JP] SynthIDプラグインスロット
│   ├── 📂 axis_semantic/        # [KR] 3축: 의미·물리 / [EN] Axis 3: Semantic & Physics / [JP] 第3軸: 意味・物理
│   │   ├── distribution.py      # [KR] CLIP 분포 편차(MPFT/TAM) / [EN] Distributional deviations / [JP] 分布偏差
│   │   ├── nsg.py               # [KR] 시공간 그래디언트(NSG) / [EN] Spatiotemporal gradient / [JP] 時空間勾配
│   │   └── consistency.py       # [KR] 프레임 일관성(DeCoF) / [EN] Frame consistency / [JP] フレーム一貫性
│   ├── 📂 fusion/
│   │   └── specsem.py           # [KR] 게이트 융합(SpecSem-Net식) / [EN] Gated fusion / [JP] ゲート融合
│   └── ensemble.py              # [KR] 동적 가중 앙상블 + 조기 확정 / [EN] Dynamic ensemble + early exit / [JP] 動的アンサンブル+早期確定
│
├── 📂 preprocessing/
│   ├── ingest.py                # [KR] 투트랙 입력 어댑터(이미지/영상) / [EN] Two-track input adapter / [JP] 2トラック入力アダプタ
│   ├── youtube.py               # [KR] yt-dlp 다운로드 + 채널 ID 추출 / [EN] yt-dlp + channel ID / [JP] yt-dlp+チャンネルID抽出
│   ├── sampler.py               # [KR] 다중 프레임 샘플링(3-Point Biopsy) / [EN] Multi-frame sampling / [JP] 多フレームサンプリング
│   └── native_patch.py          # [KR] 원본 해상도 패치 추출 / [EN] Native-resolution patches / [JP] ネイティブ解像度パッチ
│
├── 📂 infrastructure/           # [KR] 설정·비동기·로깅 / [EN] Config, async, logging / [JP] 設定・非同期・ロギング
├── 📂 jobs/                     # [KR] 백그라운드 분석 작업 / [EN] Background analysis tasks / [JP] バックグラウンド分析タスク
├── 📂 storage/                  # [KR] 캐시·모델·채널 이력 DB / [EN] Cache, models, channel history DB / [JP] キャッシュ・モデル・履歴DB
├── 📂 deploy/cloudflare/        # [KR] 데모용 보안 터널 / [EN] Secure tunnel for demos / [JP] デモ用セキュアトンネル
├── 📂 weights/                  # [KR] 모델 가중치 / [EN] Model weights / [JP] モデル重み
├── app.py
└── requirements.txt

app/ (별도 디렉토리)             # [KR] Flutter 클라이언트 / [EN] Flutter client (separate dir) / [JP] Flutterクライアント(別ディレクトリ)
```

---

<br>

## 🇰🇷 한국어 (Korean)

### 1. 개요 (Overview)

**TRINITY**는 유튜브 링크 공유(Share Sheet) 또는 이미지 업로드로 전달된 미디어가 **AI로 생성·조작되었는지**를 판별하는 **3축 앙상블 탐지 시스템**입니다. 사용자는 Flutter 크로스플랫폼 앱(경량 클라이언트)으로 요청을 보내고, 분석은 Apple Silicon 맥북에서 구동되는 **셀프호스팅 엣지 서버**(FastAPI)가 수행합니다. 영상을 상용 클라우드에 업로드하지 않는 구조로 프라이버시 노출을 최소화합니다.

Sora·Veo 등 확산 모델 기반 영상 생성 기술은 인간의 시각적 인지 능력을 넘어서는 극사실적 결과물을 만들어내고 있으며, 위협 모델도 "얼굴 교체 딥페이크"에서 **"완전 생성 영상"**으로 이동했습니다. 공간 픽셀 단서에만 의존하는 단일 모델 탐지기는 압축·노이즈·리사이즈 같은 통상적 유통 과정에서 단서를 잃고, 학습에 없던 생성기(Unseen Generators) 앞에서 일반화에 실패합니다. TRINITY는 **서로 실패 모드가 겹치지 않는 세 가지 단서**를 결합해 이 한계를 돌파합니다.

추가로, 분석 결과를 **유튜브 채널 ID 단위로 누적**하여 "이 채널에서 분석된 N개 영상 중 M개가 AI 생성 의심" 같은 채널 신뢰도 맥락을 함께 제공합니다.

### 2. 시스템 파이프라인 (Pipeline)

1. **Flutter 앱**: 유튜브 공유 시트로 링크 수신, 또는 갤러리에서 이미지 업로드 → REST API 호출
2. **제어**: FastAPI + Celery + Redis 큐 기반 비동기 처리 (요청 접수 → 백그라운드 분석 → 완료 통지)
3. **수집**: 영상 트랙은 `yt-dlp` 다운로드 + 채널 ID 추출, 이미지 트랙은 파일 직수신
4. **전처리**: 전/중/후 구간 다중 프레임 샘플링(3-Point Biopsy). **강제 리사이즈 금지** — 고주파 위조 흔적 보존을 위해 원본 해상도 패치를 유지하는 Native-Scale 원칙(ICLR 2026) 적용
5. **분석**: 3축 병렬 실행 → 게이트 융합 → 위험 점수 + 판단 근거 산출
6. **저장/반환**: 결과를 채널 이력 DB에 누적하고 앱에 반환

### 3. 3축 탐지 엔진 (The 3-Axis Engine)

#### 제1축 — 신호 (Signal / Frequency)
> *"생성 모델의 업샘플링 연산은 주파수 도메인에 주기적 스펙트럼 이상을 남긴다."*

- **원리**: 픽셀 도메인에서는 보이지 않는 생성 흔적이 FFT/DCT로 주파수 도메인에 투영하면 격자 피크·스펙트럼 이상으로 드러납니다. 고대역 마스크 필터링 후 iFFT로 재구성한 **스펙트럼 잔차 맵**은 텍스처 이상과 주기적 그리드 아티팩트를 강하게 부각시킵니다.
- **구현**: ① 스펙트럼 잔차 추출, ② **EfficientNet + CBAM 공간-주파수 하이브리드 어텐션** — RGB 시각 표현과 DCT 주파수 요소를 조기 결합해 상호 보완 활용 (FaceForensics++ C23에서 ROC-AUC 0.997 보고), ③ **주파수 도메인 마스킹 학습** — 훈련 중 주파수 대역을 무작위로 가려 특정 대역 과적합을 방지하고 unseen generator 일반화를 강화, ④ FreqNet 계열의 위상+진폭 스펙트럼 합성곱 학습.
- **강건성 근거**: 2026년 강건성 정량화 연구에서 가우시안 블러가 JPEG 압축 대비 평균 30배 파괴적임이 확인됨 — 즉 압축 위주인 유튜브 재인코딩 환경에서 주파수 단서는 상당 부분 생존합니다. 저화질(업스케일성 열화) 입력에는 품질 게이팅으로 가중치를 낮춥니다.

#### 제2축 — 출처 (Provenance)
> *"사후 적발에서 선제적 출처 증명으로: 워터마크가 있으면 즉시 확정, 없으면 판단 보류."*

| 기술 | 삽입 위치 | 성격 | 내구성 |
|---|---|---|---|
| **C2PA** | 파일 메타데이터 헤더(서명된 매니페스트) | 제작자·도구·편집 이력 등 정보 풍부 | 낮음 — 재인코딩·스크린샷·플랫폼 업로드로 유실 |
| **SynthID** | 콘텐츠 자체(픽셀/스펙트로그램) | AI 생성 여부만 표시(정보 빈약) | 높음 — 압축·크롭·리사이즈·색보정 생존 |

- **커버리지**: 2026년 5월 기준 OpenAI(ChatGPT·API 이미지)가 SynthID를 채택하고 C2PA 운영위에 합류하는 등 업계 표준화가 진행 중이며, 누적 100억 개 이상의 콘텐츠에 워터마크가 적용되었습니다.
- **Early-Exit 정책**: C2PA 매니페스트 또는 SynthID 신호가 확인되면 **연산 없이 조기 확정**하여 자원을 절약합니다. 반대로 신호 부재는 "정보 없음"으로만 취급합니다 — RobustSora 벤치마크가 보여주듯, 탐지기가 워터마크 패턴을 암기하는 지름길 학습에 빠지면 워터마크 제거 공격 앞에서 무너지기 때문입니다. 음성 판정 시 즉시 제1·3축으로 제어권을 넘깁니다.
- **구현 현실**: SynthID 검출은 구글 비공개 인프라가 필요하므로(제3자 독립 구현 불가), Content Detection API 얼리 액세스를 신청하고 승인 전까지는 **인터페이스만 정의된 플러그인 슬롯**으로 유지합니다. C2PA 파싱은 오픈 표준으로 직접 구현합니다.

#### 제3축 — 의미·물리 (Semantic & Physics)
> *"픽셀은 완벽해도, 의미의 분포와 물리 법칙은 속이기 어렵다."*

- **분포 편차 탐지**: 동결된 CLIP-ViT의 특징 공간에서 실제 이미지 분포로부터의 편차를 측정합니다. **MPFT(마스킹 기반 미세 튜닝) + TAM(질감 인식 마스킹)**으로 피사체의 의미 영역을 의도적으로 가려 "우주복 입은 동물 = 가짜" 같은 **의미론적 지름길 학습을 차단**하고, 포렌식 분포 특징에만 주의를 강제합니다 (GenImage 98.2%, UniversalFakeDetect 94.6% 평균 정확도 보고).
- **물리 기반 시공간 모델링 (영상 전용)**: NSG(정규화 시공간 그래디언트)로 공간 확률 그래디언트와 시간 밀도 변화의 비율을 정량화하고, 실제 영상과의 MMD(최대 평균 불일치)를 탐지 지표로 사용해 중력·반사·조명 등 **물리 법칙 위반**을 포착합니다 (NeurIPS 2025).
- **프레임 일관성 (영상 전용)**: DeCoF 계열 접근으로 인접 프레임 간 의미 정체성의 미세한 흔들림을 탐지합니다.

#### 융합 — SpecSem-Net식 게이트 결합
제1축의 고주파 스펙트럼 특징은 배경의 정상적 환경 노이즈까지 포함해 오탐의 원인이 됩니다. TRINITY는 SpecSem-Net의 **게이트 융합 메커니즘**을 차용해, 제3축의 의미론적 컨텍스트로 스펙트럼 특징을 적응적으로 변조 — 무의미한 노이즈를 걸러내고 정제된 특징만 병합합니다. 이는 단순 점수 투표가 아닌 **특징 수준의 화학적 결합**이며, 해당 구조는 Sora·Veo급 생성물 상대로 87% 이상의 정확도를 보고했습니다.

**앙상블 정책**:

| 조건 | 동작 |
|---|---|
| 제2축 양성 (워터마크/매니페스트 확인) | 즉시 확정 (Early-Exit), 제1·3축은 참고치 |
| 제2축 음성 | 제1축 + 제3축 게이트 융합이 주력 |
| 저화질/블러성 열화 입력 | 제1축 가중 하향, 제3축 가중 상향 (품질 게이팅) |
| 얼굴 유무 | 무관 — 전 축이 인물 없는 완전 생성 영상에도 동작 |

### 4. 투트랙: 이미지 + 영상 (Two-Track)

세 축 모두 본질적으로 프레임 단위 분석이므로, 코어 모듈을 100% 공유하고 입력 어댑터만 분기합니다.

| 축 | 이미지 트랙 | 영상 트랙 |
|---|---|---|
| 1축 신호 | ✅ (압축 적어 신호 선명) | ✅ 다중 프레임 집계 + 원본 해상도 패치 |
| 2축 출처 | ✅ C2PA + SynthID | ⚠️ 재인코딩 시 C2PA 유실 가능, SynthID는 생존 |
| 3축 의미 | ✅ 분포 편차 | ✅ 분포 편차 + NSG + 프레임 일관성 |

### 5. 벤치마크 검증 체계 (Benchmarks)

| 벤치마크 | 규모/특징 | 검증 목적 |
|---|---|---|
| **GenVidBench** (2025.1) | Sora 포함 11개 SOTA 생성기, 14.3만 영상, 교차 소스·교차 생성기 | 일반화 성능의 기준점 |
| **GenVideo / DeMamba** (2024.5) | 100만+ 생성 영상, T2V·I2V 포괄 | 유통 압축·화질 열화 저항성 |
| **RobustSora** (2025.12) | 워터마크 제거(De-watermarked) 영상 중심 | 워터마크 회피 공격 하 제1·3축 독립 성능 분리 검증 |

**일반화 → 압축 저항 → 워터마크 회피 공격**으로 이어지는 3단 검증 체계이며, DeMamba의 플러그 앤 플레이 Mamba 모듈은 경량 시간 특성 추출 레이어로 참고합니다.

### 6. 서버 하드웨어 전략 (Apple Silicon Edge Server)

분석 서버는 Apple Silicon 맥북(현재 MacBook Air 기반 개발)에서 구동되며, 연산 특성별로 이기종 코어를 분할 사용합니다.

- **FFT/주파수 변환 → CPU(AMX)·커스텀 Metal 커널**: PyTorch `torch.fft`는 MPS 백엔드에서 2의 거듭제곱 텐서 제약과 복소수 처리 이슈가 보고되어, 주파수 연산은 CPU 행렬 보조프로세서 경로로 우회합니다.
- **출처 스캐닝 → 효율 코어**: C2PA 파싱·해시 검증은 딥러닝이 아닌 암호학적 연산이므로 효율 코어 백그라운드 스레드에서 파일 로딩과 동시 수행합니다.
- **ViT/CNN 추론 → CoreML(ANE)**: 대형 비전 모델은 FP16 정적 양자화 후 Neural Engine에 할당 — MPS 대비 약 1.5배 속도 향상이 보고된 경로입니다.
- **열 관리**: 팬리스 구조의 열 스로틀링을 감안한 분할 실행으로 안정적 처리량을 유지하고, 데모 시에는 Cloudflare Tunnel로 외부 접근을 제공합니다.

### 7. Flutter 클라이언트 & 접근성 (Client & Accessibility)

- **진입**: OS 공유 시트("TRINITY로 분석") 또는 앱 내 링크 붙여넣기/이미지 업로드
- **화면 흐름**: 홈 → 분석 중 → 결과(점수+근거+채널 요약) → 채널 상세 — 탭바 없는 스택 네비게이션
- **디자인**: Bold Typography 다크 테마 (배경 `#0A0A0A`, 버밀리언 액센트 `#FF3D00`, Inter Tight + JetBrains Mono) — 판독 점수를 화면을 지배하는 헤드라인으로 제시
- **신호등 메타포**: 초록(낮은 위험) / 노랑(주의 — 출처 미확인 등) / 빨강(높은 위험)의 3단계 위험도 + 축별 근거 칩으로 복잡한 신호 분석을 직관화
- **취약 계층 배려**: 고대비 팔레트, 동적 글꼴 크기, 화면 낭독기(VoiceOver/TalkBack) 연동, 판독 결과의 TTS 음성 안내("이 영상은 AI 생성 확률이 매우 높습니다. 송금을 중단하십시오") — 딥페이크 금융 사기의 주요 표적인 노년층의 골든타임 확보를 목표로 합니다

### 8. 한계 및 주의 (Limitations)

- 제2축 음성은 **진본 증명이 아닙니다** — 워터마크 미적용 생성기가 다수이며, 악의적 제거 가능성을 기본 전제로 합니다.
- 의미론 기반 탐지는 지름길 학습 위험이 보고되어 있어 **제1축과의 앙상블을 전제**로만 동작합니다.
- NSG 등 일부 모듈은 연산 비용이 높아 품질·길이 조건에 따라 게이팅됩니다.
- 본 시스템은 **위험도 점수**를 제시하며 단정적 판정을 내리지 않습니다.
- `yt-dlp` 기반 다운로드는 플랫폼 약관상 회색지대로, 본 프로젝트는 **학술 연구 목적**으로만 사용합니다.

---

<br>
## 🇺🇸 English

### 1. Overview

**TRINITY** is a **3-axis ensemble detection system** that determines whether media shared via a YouTube link (Share Sheet) or an uploaded image was **generated or manipulated by AI**. Users interact through a lightweight Flutter cross-platform app, while analysis runs on a **self-hosted edge server** (FastAPI) on an Apple Silicon MacBook — media never leaves for a commercial cloud, minimizing privacy exposure.

Diffusion-based generators such as Sora and Veo now produce hyper-realistic footage beyond human perceptual limits, and the threat model has shifted from face-swap deepfakes to **fully generated video**. Single-model detectors that rely on spatial pixel cues lose their evidence under routine distribution transforms (compression, noise, resizing) and fail to generalize to unseen generators. TRINITY overcomes this by combining **three cues with non-overlapping failure modes**.

Results are also **aggregated per YouTube channel ID**, providing context such as "M of N analyzed videos from this channel were flagged as likely AI-generated."

### 2. System Pipeline

1. **Flutter app**: receives a YouTube link via the OS share sheet, or an image upload → calls the REST API
2. **Control**: FastAPI + Celery + Redis queue-based async processing (accept → background analysis → notify)
3. **Ingestion**: video track via `yt-dlp` download + channel ID extraction; image track via direct file upload
4. **Preprocessing**: multi-frame sampling across start/middle/end segments (3-Point Biopsy). **No forced resizing** — native-resolution patches are preserved to protect high-frequency forgery traces (Native-Scale principle, ICLR 2026)
5. **Analysis**: three axes run in parallel → gated fusion → risk score + evidence
6. **Store/return**: results accumulate in the channel-history DB and return to the app

### 3. The 3-Axis Detection Engine

#### Axis 1 — Signal (Frequency)
> *"Generative up-sampling leaves periodic spectral anomalies in the frequency domain."*

- **Principle**: traces invisible in pixel space emerge as grid peaks and spectral anomalies when projected via FFT/DCT. A **spectral residual map** — high-pass mask filtering followed by iFFT reconstruction — strongly amplifies texture anomalies and periodic grid artifacts.
- **Implementation**: ① spectral residual extraction; ② a **hybrid spatial-frequency attention model (EfficientNet + CBAM)** that fuses RGB representations with DCT components early, reported at ROC-AUC 0.997 on FaceForensics++ C23; ③ **frequency-domain masking** during training — randomly masking frequency bands to prevent over-fitting to any single band and to improve generalization to unseen generators; ④ FreqNet-style convolution on both phase and amplitude spectra.
- **Robustness evidence**: a 2026 quantitative study found Gaussian blur roughly 30× more destructive than JPEG compression — meaning frequency cues largely survive YouTube's compression-centric re-encoding. Low-quality (blur-like degraded) inputs are down-weighted via quality gating.

#### Axis 2 — Provenance
> *"From post-hoc detection to proactive provenance: a watermark confirms instantly; its absence proves nothing."*

| Technology | Embedded in | Character | Durability |
|---|---|---|---|
| **C2PA** | file metadata header (signed manifest) | information-rich: creator, tool, edit history | Low — lost to re-encoding, screenshots, platform uploads |
| **SynthID** | the content itself (pixels/spectrogram) | information-poor: AI-origin signal only | High — survives compression, cropping, resizing, color edits |

- **Coverage**: as of May 2026, OpenAI adopted SynthID for ChatGPT/API images and joined the C2PA steering committee, with 10B+ pieces of content watermarked to date — an emerging industry standard.
- **Early-Exit policy**: if a C2PA manifest or SynthID signal is verified, TRINITY **confirms immediately without further compute**. Conversely, a negative result is treated strictly as "no information" — as the RobustSora benchmark showed, detectors that memorize watermark patterns (shortcut learning) collapse under watermark-removal attacks. On a negative, control passes straight to Axes 1 and 3.
- **Implementation reality**: SynthID detection requires Google's proprietary infrastructure (no independent third-party implementation), so TRINITY registers for the Content Detection API early access and keeps the module as an **interface-defined plugin slot** until approved. C2PA parsing is implemented directly as an open standard.

#### Axis 3 — Semantic & Physics
> *"Pixels can be perfect; semantic distributions and physical laws are harder to fake."*

- **Distributional deviation detection**: measures deviation from the real-image distribution in a frozen CLIP-ViT feature space. **MPFT (masking-based fine-tuning) + TAM (texture-aware masking)** deliberately mask semantic subject regions to **block semantic-shortcut learning** ("astronaut animal = fake") and force attention onto forensic distributional features (reported 98.2% on GenImage, 94.6% on UniversalFakeDetect).
- **Physics-driven spatiotemporal modeling (video only)**: NSG (Normalized Spatiotemporal Gradient) quantifies the ratio of spatial probability gradients to temporal density changes, using MMD against real-video statistics to catch **violations of physical dynamics** — gravity, reflections, lighting (NeurIPS 2025).
- **Frame consistency (video only)**: DeCoF-style detection of subtle identity jitter between adjacent frames.

#### Fusion — SpecSem-Net-style gated merging
High-frequency spectral features inevitably include benign environmental noise, causing false positives. TRINITY adopts SpecSem-Net's **gated merging mechanism**: semantic context from Axis 3 adaptively modulates the spectral features from Axis 1, filtering out meaningless noise before merging. This is **feature-level fusion**, not naive score voting; the architecture reported 87%+ accuracy against Sora/Veo-class generators.

**Ensemble policy**:

| Condition | Behavior |
|---|---|
| Axis 2 positive (watermark/manifest verified) | Immediate confirmation (early exit); Axes 1 & 3 advisory |
| Axis 2 negative | Gated fusion of Axes 1 + 3 takes over |
| Low-quality / blur-degraded input | Axis 1 down-weighted, Axis 3 up-weighted (quality gating) |
| Face presence | Irrelevant — all axes work on fully generated, faceless footage |

### 4. Two-Track: Image + Video

All three axes are fundamentally frame-level analyses, so core modules are 100% shared with only the input adapter branching.

| Axis | Image track | Video track |
|---|---|---|
| 1 Signal | ✅ (cleaner signal, less compression) | ✅ multi-frame aggregation + native-resolution patches |
| 2 Provenance | ✅ C2PA + SynthID | ⚠️ C2PA may be stripped by re-encoding; SynthID survives |
| 3 Semantic | ✅ distributional deviation | ✅ deviation + NSG + frame consistency |

### 5. Benchmark Validation

| Benchmark | Scale / Features | Purpose |
|---|---|---|
| **GenVidBench** (2025.1) | 11 SOTA generators incl. Sora, 143K videos, cross-source & cross-generator | Baseline for generalization |
| **GenVideo / DeMamba** (2024.5) | 1M+ generated videos, T2V & I2V | Resistance to distribution compression/degradation |
| **RobustSora** (2025.12) | De-watermarked videos | Isolated validation of Axes 1 & 3 under watermark-removal attacks |

A three-stage regime — **generalization → compression resistance → watermark-evasion attack** — with DeMamba's plug-and-play Mamba module referenced for lightweight temporal feature extraction.

### 6. Server Hardware Strategy (Apple Silicon Edge Server)

The analysis server runs on an Apple Silicon MacBook (currently developed on a MacBook Air), splitting workloads across heterogeneous cores:

- **FFT/frequency transforms → CPU (AMX) / custom Metal kernels**: PyTorch `torch.fft` on the MPS backend has reported power-of-two tensor constraints and complex-type issues, so frequency ops route through the CPU matrix coprocessor path.
- **Provenance scanning → efficiency cores**: C2PA parsing and hash verification are cryptographic, not neural, so they run on efficiency-core background threads concurrently with file loading.
- **ViT/CNN inference → CoreML (ANE)**: large vision models are statically quantized to FP16 and assigned to the Neural Engine — a path reported at ~1.5× speedup over MPS.
- **Thermals**: split execution accounts for fanless throttling to sustain throughput; demos are exposed via Cloudflare Tunnel.

### 7. Flutter Client & Accessibility

- **Entry**: OS share sheet ("Analyze with TRINITY") or in-app link paste / image upload
- **Flow**: Home → Analyzing → Result (score + evidence + channel summary) → Channel Detail — stack navigation, no tab bar
- **Design**: Bold Typography dark theme (background `#0A0A0A`, vermillion accent `#FF3D00`, Inter Tight + JetBrains Mono) — the verdict score presented as a screen-dominating headline
- **Traffic-light metaphor**: green (low risk) / yellow (caution — e.g., provenance unverified) / red (high risk), with per-axis evidence chips translating complex signal analysis into intuition
- **Accessibility for vulnerable users**: high-contrast palette, dynamic font scaling, screen-reader (VoiceOver/TalkBack) integration, and TTS verdicts ("This video is very likely AI-generated. Stop any money transfer now.") — aimed at protecting older adults, the primary targets of deepfake financial fraud, within the golden hour

### 8. Limitations

- An Axis-2 negative is **not proof of authenticity** — many generators embed no watermark, and malicious stripping is assumed by default.
- Semantic detection carries documented shortcut-learning risks and therefore operates **only in ensemble with Axis 1**.
- Compute-heavy modules (e.g., NSG) are gated by input quality and length.
- The system reports a **risk score** and never issues a categorical verdict.
- `yt-dlp` downloads sit in a platform-ToS gray area; this project is **for academic research only**.

---

<br>
## 🇯🇵 日本語 (Japanese)

### 1. 概要 (Overview)

**TRINITY**は、YouTubeリンクの共有(Share Sheet)または画像アップロードで届いたメディアが**AIによって生成・操作されたか**を判別する**3軸アンサンブル検知システム**です。ユーザーは軽量なFlutterクロスプラットフォームアプリで操作し、分析はApple Silicon MacBook上で動作する**セルフホスト型エッジサーバー**(FastAPI)が実行します。メディアを商用クラウドへ送信しない構成により、プライバシー露出を最小化します。

SoraやVeoに代表される拡散モデル系の映像生成技術は人間の知覚限界を超える超写実的な結果を生み出しており、脅威モデルも「顔交換ディープフェイク」から**「完全生成映像」**へと移行しました。空間ピクセルの手掛かりだけに依存する単一モデル検知器は、圧縮・ノイズ・リサイズといった通常の流通過程で証拠を失い、学習に含まれない未知の生成器(Unseen Generators)への一般化に失敗します。TRINITYは**互いに失敗モードが重ならない3つの手掛かり**を結合してこの限界を突破します。

さらに分析結果を**YouTubeチャンネルID単位で蓄積**し、「このチャンネルで分析されたN本中M本がAI生成の疑い」といったチャンネル信頼度の文脈も提供します。

### 2. システムパイプライン (Pipeline)

1. **Flutterアプリ**: 共有シートからYouTubeリンクを受信、または画像をアップロード → REST APIを呼び出し
2. **制御**: FastAPI + Celery + Redis のキューによる非同期処理(受付 → バックグラウンド分析 → 完了通知)
3. **取得**: 映像トラックは`yt-dlp`でダウンロードしチャンネルIDを抽出、画像トラックはファイル直接受信
4. **前処理**: 冒頭・中盤・終盤の多フレームサンプリング(3-Point Biopsy)。**強制リサイズ禁止** — 高周波の偽造痕跡を守るため、ネイティブ解像度パッチを維持するNative-Scale原則(ICLR 2026)を適用
5. **分析**: 3軸を並列実行 → ゲート融合 → リスクスコア + 判断根拠を算出
6. **保存/返却**: 結果をチャンネル履歴DBへ蓄積し、アプリへ返却

### 3. 3軸検知エンジン (The 3-Axis Engine)

#### 第1軸 — 信号 (Signal / Frequency)
> *「生成モデルのアップサンプリング演算は、周波数ドメインに周期的なスペクトル異常を残す。」*

- **原理**: ピクセル空間では見えない生成痕跡が、FFT/DCTで周波数ドメインへ投影すると格子状ピークやスペクトル異常として現れます。高域マスクフィルタリング後にiFFTで再構成した**スペクトル残差マップ**は、テクスチャ異常と周期的グリッドアーティファクトを強く浮かび上がらせます。
- **実装**: ① スペクトル残差抽出、② **EfficientNet + CBAM 空間-周波数ハイブリッドアテンション** — RGB表現とDCT周波数要素を早期結合し相互補完的に活用(FaceForensics++ C23でROC-AUC 0.997の報告)、③ **周波数ドメインマスキング学習** — 訓練中に周波数帯域をランダムに遮蔽し、特定帯域への過学習を防いで未知生成器への一般化を強化、④ FreqNet系の位相+振幅スペクトルへの畳み込み学習。
- **頑健性の根拠**: 2026年の定量研究で、ガウシアンブラーがJPEG圧縮の約30倍破壊的であることが確認されました — つまり圧縮中心のYouTube再エンコード環境では周波数の手掛かりは相当程度生存します。低画質(ブラー性劣化)入力には品質ゲーティングで重みを下げます。

#### 第2軸 — 出所 (Provenance)
> *「事後の摘発から先制的な出所証明へ: 透かしがあれば即時確定、なければ判断保留。」*

| 技術 | 埋め込み位置 | 性格 | 耐久性 |
|---|---|---|---|
| **C2PA** | ファイルメタデータヘッダ(署名付きマニフェスト) | 制作者・ツール・編集履歴など情報豊富 | 低 — 再エンコード・スクリーンショット・プラットフォーム投稿で消失 |
| **SynthID** | コンテンツ自体(ピクセル/スペクトログラム) | AI生成の有無のみ(情報は乏しい) | 高 — 圧縮・クロップ・リサイズ・色調整でも生存 |

- **カバレッジ**: 2026年5月時点でOpenAIがChatGPT/API画像へSynthIDを採用しC2PA運営委員会にも参加するなど業界標準化が進行中で、累計100億件以上のコンテンツに透かしが適用されています。
- **Early-Exitポリシー**: C2PAマニフェストまたはSynthID信号が確認されれば**追加演算なしで即時確定**し、資源を節約します。逆に信号の不在は「情報なし」としてのみ扱います — RobustSoraベンチマークが示した通り、透かしパターンを暗記するショートカット学習に陥った検知器は、透かし除去攻撃の前に崩壊するためです。陰性の場合は直ちに第1・3軸へ制御を渡します。
- **実装の現実**: SynthID検出にはGoogleの非公開インフラが必要なため(第三者による独立実装は不可)、Content Detection APIのアーリーアクセスを申請し、承認まではモジュールを**インターフェース定義済みのプラグインスロット**として保持します。C2PA解析はオープン標準として直接実装します。

#### 第3軸 — 意味・物理 (Semantic & Physics)
> *「ピクセルは完璧でも、意味の分布と物理法則を偽るのは難しい。」*

- **分布偏差検知**: 凍結したCLIP-ViTの特徴空間で、実画像分布からの偏差を測定します。**MPFT(マスキングベース微調整)+ TAM(テクスチャ認識マスキング)**で被写体の意味領域を意図的に遮蔽し、「宇宙服の動物=偽物」のような**意味的ショートカット学習を遮断**、フォレンジックな分布特徴のみに注意を強制します(GenImage 98.2%、UniversalFakeDetect 94.6%の平均精度を報告)。
- **物理ベース時空間モデリング(映像のみ)**: NSG(正規化時空間勾配)で空間確率勾配と時間密度変化の比率を定量化し、実映像統計とのMMD(最大平均不一致)を検知指標として、重力・反射・照明などの**物理法則違反**を捕捉します(NeurIPS 2025)。
- **フレーム一貫性(映像のみ)**: DeCoF系のアプローチで、隣接フレーム間の意味的アイデンティティの微細な揺らぎを検知します。

#### 融合 — SpecSem-Net式ゲート結合
第1軸の高周波スペクトル特徴は背景の正常な環境ノイズまで含み、誤検知の原因となります。TRINITYはSpecSem-Netの**ゲート融合メカニズム**を採用し、第3軸の意味的コンテキストでスペクトル特徴を適応的に変調 — 無意味なノイズを濾過し、精製された特徴のみを併合します。これは単純なスコア投票ではなく**特徴レベルの化学的結合**であり、この構造はSora・Veo級の生成物に対して87%以上の精度を報告しています。

**アンサンブルポリシー**:

| 条件 | 動作 |
|---|---|
| 第2軸 陽性(透かし/マニフェスト確認) | 即時確定(Early-Exit)、第1・3軸は参考値 |
| 第2軸 陰性 | 第1軸+第3軸のゲート融合が主力 |
| 低画質/ブラー性劣化入力 | 第1軸の重みを下げ、第3軸を上げる(品質ゲーティング) |
| 顔の有無 | 無関係 — 全軸が人物のいない完全生成映像にも動作 |

### 4. 2トラック: 画像 + 映像 (Two-Track)

3軸はいずれも本質的にフレーム単位の分析であるため、コアモジュールを100%共有し、入力アダプタのみ分岐します。

| 軸 | 画像トラック | 映像トラック |
|---|---|---|
| 第1軸 信号 | ✅(圧縮が少なく信号鮮明) | ✅ 多フレーム集計 + ネイティブ解像度パッチ |
| 第2軸 出所 | ✅ C2PA + SynthID | ⚠️ 再エンコードでC2PA消失の可能性、SynthIDは生存 |
| 第3軸 意味 | ✅ 分布偏差 | ✅ 分布偏差 + NSG + フレーム一貫性 |

### 5. ベンチマーク検証体系 (Benchmarks)

| ベンチマーク | 規模/特徴 | 検証目的 |
|---|---|---|
| **GenVidBench** (2025.1) | Sora含む11のSOTA生成器、14.3万本、クロスソース・クロス生成器 | 一般化性能の基準点 |
| **GenVideo / DeMamba** (2024.5) | 100万本以上の生成映像、T2V・I2V包括 | 流通圧縮・画質劣化への耐性 |
| **RobustSora** (2025.12) | 透かし除去(De-watermarked)映像中心 | 透かし回避攻撃下での第1・3軸の独立性能を分離検証 |

**一般化 → 圧縮耐性 → 透かし回避攻撃**と続く3段階の検証体系であり、DeMambaのプラグアンドプレイ型Mambaモジュールは軽量な時間特徴抽出レイヤーとして参照します。

### 6. サーバーハードウェア戦略 (Apple Silicon Edge Server)

分析サーバーはApple Silicon MacBook(現在はMacBook Airで開発)上で動作し、演算特性ごとに異種コアを分割使用します。

- **FFT/周波数変換 → CPU(AMX)・カスタムMetalカーネル**: PyTorch `torch.fft`はMPSバックエンドで2のべき乗テンソル制約や複素数処理の問題が報告されており、周波数演算はCPU行列コプロセッサ経路で迂回します。
- **出所スキャン → 効率コア**: C2PA解析・ハッシュ検証はニューラルではなく暗号学的演算のため、効率コアのバックグラウンドスレッドでファイル読み込みと同時に実行します。
- **ViT/CNN推論 → CoreML(ANE)**: 大型ビジョンモデルはFP16静的量子化の上でNeural Engineに割り当て — MPS比で約1.5倍の高速化が報告されている経路です。
- **熱管理**: ファンレス構造のサーマルスロットリングを考慮した分割実行で安定したスループットを維持し、デモ時はCloudflare Tunnelで外部アクセスを提供します。

### 7. Flutterクライアント & アクセシビリティ (Client & Accessibility)

- **入口**: OS共有シート(「TRINITYで分析」)またはアプリ内リンク貼り付け/画像アップロード
- **画面フロー**: ホーム → 分析中 → 結果(スコア+根拠+チャンネル要約) → チャンネル詳細 — タブバーなしのスタックナビゲーション
- **デザイン**: Bold Typographyダークテーマ(背景 `#0A0A0A`、バーミリオンアクセント `#FF3D00`、Inter Tight + JetBrains Mono) — 判定スコアを画面を支配する見出しとして提示
- **信号機メタファー**: 緑(低リスク)/ 黄(注意 — 出所未確認など)/ 赤(高リスク)の3段階リスク表示 + 軸別根拠チップで、複雑な信号分析を直観化
- **デジタル弱者への配慮**: 高コントラストパレット、動的フォントサイズ、スクリーンリーダー(VoiceOver/TalkBack)連携、判定結果のTTS音声案内(「この映像はAI生成の可能性が非常に高いです。送金を直ちに中止してください」)— ディープフェイク金融詐欺の主要標的である高齢者のゴールデンタイム確保を目指します

### 8. 制限と注意 (Limitations)

- 第2軸の陰性は**真正の証明ではありません** — 透かし未適用の生成器が多数存在し、悪意ある除去の可能性を基本前提とします。
- 意味ベースの検知にはショートカット学習のリスクが報告されており、**第1軸とのアンサンブルを前提**としてのみ動作します。
- NSGなど一部モジュールは演算コストが高く、品質・長さ条件によりゲーティングされます。
- 本システムは**リスクスコア**を提示するものであり、断定的判定は行いません。
- `yt-dlp`によるダウンロードはプラットフォーム規約上グレーゾーンであり、本プロジェクトは**学術研究目的**に限定して使用します。

---

## 📚 References (2024–2026)

### Axis 1 — Signal / Frequency
- *Frequency-Aware Robustness Analysis of Deepfake Detection Models* — Journal on Artificial Intelligence, 2026.3 — [techscience.com/jai/v8n1/66558](https://www.techscience.com/jai/v8n1/66558/html)
- *A Hybrid Spatial–Frequency Attention-Based Algorithm Using EfficientNet for Robust and Interpretable Deepfake Detection* — Scientific Reports, 2026.4 — [nature.com/articles/s41598-026-46086-9](https://www.nature.com/articles/s41598-026-46086-9)
- *Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking* — ACM TOMMCCAP, 2026.3 — [doi.org/10.1145/3797266](https://doi.org/10.1145/3797266)
- *Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale* — ICLR 2026 — [openreview.net/forum?id=XD431fRCg6](https://openreview.net/forum?id=XD431fRCg6)
- *FreqNet: Frequency-Aware Deepfake Detection* — AAAI 2024 — [arXiv:2403.07240](https://arxiv.org/abs/2403.07240)

### Axis 2 — Provenance
- *Google Expands SynthID Adoption, Previews Content Detection API* — InfoQ, 2026.5 — [infoq.com](https://www.infoq.com/news/2026/05/google-synthid-content-detection/)
- *SynthID Detector — a new portal to help identify AI-generated content* — Google DeepMind — [blog.google](https://blog.google/innovation-and-ai/products/google-synthid-ai-content-detector/)
- *RobustSora: De-Watermarked Benchmark for Robust AI-Generated Video Detection* — 2025.12 — [arXiv:2512.10248](https://arxiv.org/abs/2512.10248)
- *C2PA — Coalition for Content Provenance and Authenticity* — [c2pa.org](https://c2pa.org/)

### Axis 3 — Semantic & Physics
- *Detecting AI-Generated Images via Distributional Deviations from Real Images* — 2026.1 — [arXiv:2601.03586](https://arxiv.org/abs/2601.03586)
- *When Detectors Forget Forensics: Blocking Semantic Shortcuts for Generalizable AI-Generated Image Detection* — 2026.3 — [arXiv:2603.09242](https://arxiv.org/abs/2603.09242)
- *Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection (NSG-VD)* — NeurIPS 2025
- *DeCoF: Generated Video Detection via Frame Consistency* — 2024 — [arXiv:2402.02085](https://arxiv.org/abs/2402.02085)

### Fusion
- *SpecSem-Net: Integrating Spectral and Semantic Features for Robust AI-Generated Video Detection* — 2026.5 — [arXiv:2605.17311](https://arxiv.org/abs/2605.17311)

### Benchmarks
- *GenVidBench: A Challenging Benchmark for Detecting AI-Generated Video* — 2025.1 — [arXiv:2501.11340](https://arxiv.org/abs/2501.11340)
- *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark* — 2024.5 — [arXiv:2405.19707](https://arxiv.org/abs/2405.19707)

---

> 🎓 **Academic Research Project** — Undergraduate graduation project, Dept. of Computer Engineering.
> This system reports probabilistic risk scores for research and education purposes and must not be used as sole evidence for legal or financial decisions.
