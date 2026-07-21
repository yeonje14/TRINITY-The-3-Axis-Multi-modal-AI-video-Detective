# 🛡️ TRINITY v4: 3-Axis Ensemble AI-Generated Media Detector

"신호가 말하고, 출처가 증언하며, 의미가 검증한다."  
"Signals speak. Provenance testifies. Semantics verify."  
「信号が語り、出所が証言し、意味が検証する。」

🌍 **Language Select**  
🇰🇷 한국어 (Korean) | 🇺🇸 English | 🇯🇵 日本語 (Japanese)

⚠️ **v4 (2026-07) — 선행연구 검증 반영 / Research Prototype**  
2024–2026 문헌 20여 편의 교차 검증(`docs/literature-review/` 참조) 결과를 설계에 반영했습니다: ① 제2축 Early-Exit 정책 폐기·완화 (mark-to-frame 스푸핑 실증에 대응) ② 무결성 충돌 교차 감사(cross-layer audit) 모듈 신설 ③ 영상/이미지 투트랙 대칭형 학습 데이터 파이프라인 명문화 ④ 학습 데이터 위생(지름길 학습 3종 차단) 명문화 ⑤ 참고문헌 전면 갱신.

---

## 📂 Project Structure

```text
TRINITY/
├── 📂 interfaces/
│   └── 📂 api/                  # [KR] Flutter용 REST API / [EN] REST API for Flutter client
│       ├── routes.py            # [KR] 엔드포인트 / [EN] Endpoints
│       └── schemas.py           # [KR] 요청·응답 스키마 / [EN] Request/response schemas
│
├── 📂 core/                     # [KR] 3축 탐지 엔진 / [EN] 3-Axis Detection Engine
│   ├── 📂 axis_signal/          # [KR] 1축: 신호(주파수) / [EN] Axis 1: Signal (Frequency)
│   │   ├── spectrum.py          # [KR] FFT/DCT 스펙트럼 잔차
│   │   ├── hybrid_attention.py  # [KR] 공간-주파수 하이브리드(EfficientNet+CBAM)
│   │   └── freq_masking.py      # [KR] 주파수 마스킹 학습
│   ├── 📂 axis_provenance/      # [KR] 2축: 출처 증명 / [EN] Axis 2: Provenance
│   │   ├── c2pa.py              # [KR] C2PA 매니페스트 검증
│   │   ├── synthid.py           # [KR] SynthID 플러그인 슬롯
│   │   └── audit.py             # [KR] 무결성 충돌 교차 감사 (v4 신설)
│   ├── 📂 axis_semantic/        # [KR] 3축: 의미·물리 / [EN] Axis 3: Semantic & Physics
│   │   ├── distribution.py      # [KR] 분포 편차 + 의미 감산
│   │   ├── nsg.py               # [KR] 시공간 그래디언트(NSG)
│   │   └── consistency.py       # [KR] 프레임 일관성(DeCoF)
│   ├── 📂 fusion/
│   │   └── specsem.py           # [KR] 게이트 융합(SpecSem-Net식)
│   └── ensemble.py              # [KR] 동적 가중 앙상블
│
├── 📂 preprocessing/
│   ├── ingest.py                # [KR] 투트랙 입력 어댑터(이미지/영상)
│   ├── youtube.py               # [KR] yt-dlp 다운로드 + 채널 ID 추출
│   ├── sampler.py               # [KR] 다중 프레임 샘플링(3-Point Biopsy)
│   └── native_patch.py          # [KR] 원본 해상도 패치 추출
│
├── 📂 training/                 # [KR] 학습 파이프라인 + 데이터 위생 (v4)
│   └── hygiene.py               # [KR] 압축 정렬·워터마크 탈상관·의미 감산
│
├── 📂 infrastructure/           # [KR] 설정·비동기·로깅
├── 📂 jobs/                     # [KR] 백그라운드 분석 작업
├── 📂 storage/                  # [KR] 캐시·모델·채널 이력 DB
├── 📂 deploy/cloudflare/        # [KR] 데모용 보안 터널
├── 📂 docs/literature-review/   # [KR] 선행연구 검토(검증판)
├── 📂 weights/                  # [KR] 모델 가중치
├── app.py
└── requirements.txt

app/ (별도 디렉토리)             # [KR] Flutter 클라이언트 / [EN] Flutter client

## 🇰🇷 한국어 (Korean)

### 1. 개요 (Overview)
TRINITY는 유튜브 링크 공유(Share Sheet) 또는 이미지 업로드로 전달된 미디어가 AI로 생성·조작되었는지를 판별하는 **3축 앙상블 탐지 시스템**입니다. 사용자는 Flutter 크로스플랫폼 앱(경량 클라이언트)으로 요청을 보내고, 분석은 Apple Silicon 맥북에서 구동되는 셀프호스팅 엣지 서버(FastAPI)가 수행합니다. 영상을 상용 클라우드에 업로드하지 않는 구조로 프라이버시 노출을 최소화합니다.

Sora·Veo 등 확산 모델 기반 영상 생성 기술은 인간의 시각적 인지 능력을 넘어서는 극사실적 결과물을 만들어내고 있으며, 위협 모델도 "얼굴 교체 딥페이크"에서 **"완전 생성 영상"**으로 이동했습니다. 최신 서베이(ACL Findings 2026)는 이 변화를 "아티팩트 매칭 → 사실적 충실도 검증(Factual Fidelity Verification)"의 패러다임 전환으로 정의합니다. 공간 픽셀 단서에만 의존하는 단일 모델 탐지기는 압축·노이즈·리사이즈 같은 통상적 유통 과정에서 단서를 잃고, 일반화에 실패합니다. TRINITY는 서로 실패 모드가 겹치지 않는 세 가지 단서를 결합해 이 한계를 돌파합니다.

추가로, 분석 결과를 유튜브 채널 ID 단위로 누적하여 "이 채널에서 분석된 N개 영상 중 M개가 AI 생성 의심" 같은 채널 신뢰도 맥락을 함께 제공합니다.

### 2. 시스템 파이프라인 (Pipeline)
* **Flutter 앱:** 유튜브 공유 시트로 링크 수신, 또는 갤러리에서 이미지 업로드 → REST API 호출
* **제어:** FastAPI + Celery + Redis 큐 기반 비동기 처리
* **수집:** 영상 트랙은 `yt-dlp` 다운로드 + 채널 ID 추출, 이미지 트랙은 파일 직수신
* **전처리:** 전/중/후 구간 다중 프레임 샘플링(3-Point Biopsy). **강제 리사이즈 금지** — 고주파 위조 흔적 보존을 위해 원본 해상도 패치를 유지하는 Native-Scale 원칙(ICLR 2026) 적용
* **분석:** 3축 병렬 실행 → 무결성 충돌 교차 감사 → 게이트 융합 → 위험 점수 + 판단 근거 산출
* **저장/반환:** 결과를 채널 이력 DB에 누적하고 앱에 반환

### 3. 3축 탐지 엔진 (The 3-Axis Engine)

#### 제1축 — 신호 (Signal / Frequency)
> "생성 모델의 업샘플링 연산은 주파수 도메인에 주기적 스펙트럼 이상을 남긴다."

* **원리:** 진짜/생성물 간 불일치가 공간 도메인보다 주파수 도메인에서 증폭됩니다(귀납적 편향 관점). 픽셀 도메인에서는 보이지 않는 흔적이 FFT/DCT로 투영하면 격자 피크로 드러나며, 고대역 마스크 필터링 후 iFFT로 재구성한 스펙트럼 잔차 맵은 텍스처 이상을 강하게 부각시킵니다.
* **구현:** ① 스펙트럼 잔차 추출, ② EfficientNet + CBAM 공간-주파수 하이브리드 어텐션, ③ 주파수 도메인 마스킹 학습, ④ FreqNet 계열 합성곱 학습.
* **강건성:** 2026년 정량화 연구에 따라 압축 위주인 유튜브 재인코딩 환경에서도 주파수 단서는 상당 부분 생존합니다. 저화질 입력에는 품질 게이팅으로 가중치를 낮춥니다.

#### 제2축 — 출처 (Provenance)
> "양성은 강한 증거, 음성은 무정보 — 그리고 양성조차 절대적이지 않다."

| 기술 | 삽입 위치 | 성격 | 내구성 |
| :--- | :--- | :--- | :--- |
| **C2PA** | 파일 메타데이터 헤더 | 정보 풍부 | **낮음** — 재인코딩·스크린샷으로 유실 |
| **SynthID류** | 콘텐츠 자체(픽셀) | AI 생성 여부만 표시 | **높음** — 압축·크롭 생존 |

* **판정 정책 (v4 개정):** 워터마크 신호는 **긍정 증거(positive evidence)로만** 사용합니다. 확산 정제 공격 등으로 상용 워터마크도 세탁될 수 있음이 증명되었습니다. 기존 v2의 "양성 → 즉시 확정(Early-Exit)" 정책은 폐기합니다(mark-to-frame 스푸핑 실증). 양성 시 제1·3축의 경량 교차 확인을 반드시 병행합니다.
* **무결성 충돌 교차 감사 (`audit.py`):** "사람이 편집함"을 주장하는 C2PA 매니페스트와 "AI 생성"을 알리는 픽셀 워터마크가 공존하는 "인증된 가짜"를 탐지합니다. 두 계층의 결과를 충돌 매트릭스로 교차 감사하여 모순 시 **세탁 의심 신호로 승격**시킵니다.

#### 제3축 — 의미·물리 (Semantic & Physics)
> "픽셀은 완벽해도, 의미의 분포와 물리 법칙은 속이기 어렵다."

* **분포 편차 + 의미 감산:** 실제 이미지 분포로부터의 편차를 측정하되, 피사체의 의미를 판별 기준으로 암기하는 **의미론적 지름길을 수학적으로 차단**합니다(SVD 또는 마스킹 미세튜닝).
* **물리 기반 시공간 모델링 (영상 전용):** NSG(정규화 시공간 그래디언트)로 물리 법칙 위반(중력, 반사 등)을 포착합니다.
* **프레임 일관성 (영상 전용):** 인접 프레임 간 의미 정체성의 미세한 흔들림(DeCoF)을 탐지합니다.

#### 융합 — SpecSem-Net식 게이트 결합
제3축의 의미론적 컨텍스트로 제1축의 고주파 스펙트럼 특징을 적응적으로 변조하여, 무의미한 환경 노이즈를 걸러내고 정제된 특징만 병합합니다.

### 4. 투트랙: 이미지 + 영상 (Two-Track)
코어 모듈을 100% 공유하고 입력 어댑터만 분기합니다.

| 축 | 이미지 트랙 | 영상 트랙 |
| :--- | :--- | :--- |
| **1축 신호** | ✅ (압축 적어 신호 선명) | ✅ 다중 프레임 집계 + 원본 해상도 패치 |
| **2축 출처** | ✅ C2PA + 워터마크 + 교차 감사 | ⚠️ 재인코딩 시 C2PA 유실 가능, 워터마크 생존 |
| **3축 의미** | ✅ 분포 편차 + 의미 감산 | ✅ 분포 편차 + NSG + 프레임 일관성 |

### 5. 학습 데이터 및 벤치마크 (투트랙 데이터 파이프라인)
이미지와 영상 각각의 모달리티에 맞춰 3단계 강건성 데이터셋을 대칭적으로 구성했습니다.

**🎥 영상 트랙 (Video)**
* **기초 훈련:** GenVideo / DeMamba (100만+ 스케일, 화질 열화 저항성)
* **실전 평가:** GenVidBench (14.3만, 교차 생성기 일반화)
* **위생/강건성:** RobustSora (워터마크 제거 데이터, 1·3축 독립 성능 검증)

**📸 이미지 트랙 (Image)**
* **기초 훈련:** GenImage (268만 장, 분포/주파수 사전 학습)
* **실전 평가:** WildFake (실전형 야생 데이터, 일반화 검증)
* **위생/강건성:** MIRAGE (실사·생성물 동일 커뮤니티 수집, 도메인 편향 꼼수 원천 차단)

### 6. 학습 데이터 위생 — 지름길 학습 차단 (v4)
2024~2026 문헌은 세 축 모두 "가장 값싼 상관관계를 암기한다"고 경고합니다. TRINITY는 이를 `training/hygiene.py`에 명문화하여 차단합니다.

| 축 | 지름길 발생 원인 | 차단책 (근거) |
| :--- | :--- | :--- |
| **1축 신호** | 진짜(웹 재압축) vs 가짜(원시 생성)의 압축 비대칭 | DDA식 압축·스펙트럼 정렬 (NeurIPS 2025) |
| **2축 출처** | 합성물에만 워터마크가 존재하는 훈련 데이터 | 양 클래스 워터마크 탈상관 (arXiv:2606.23335) |
| **3축 의미** | 생성 데이터셋의 의미 분포 쏠림 | 의미 부분공간 감산(SVD) 또는 마스킹 미세튜닝 |

### 7. 서버 하드웨어 전략 (Apple Silicon Edge Server)
분석 서버는 Apple Silicon 맥북에서 구동되며, 연산 특성별로 이기종 코어를 분할 사용합니다.
* **FFT/주파수 변환 → CPU(AMX):** PyTorch `torch.fft`의 MPS 제약을 우회하여 보조프로세서 할당.
* **출처 스캐닝·교차 감사 → 효율 코어:** 파일 로딩과 동시 수행.
* **ViT/CNN 추론 → CoreML(ANE):** FP16 정적 양자화 후 Neural Engine 할당 (MPS 대비 속도 향상).

### 8. Flutter 클라이언트 & 접근성
* **진입:** OS 공유 시트("TRINITY로 분석") 또는 앱 내 이미지 업로드.
* **디자인:** Bold Typography 다크 테마 (배경 `#0A0A0A`, 버밀리언 액센트 `#FF3D00`).
* **취약 계층 배려:** 고대비 팔레트, 동적 글꼴 크기, 화면 낭독기(TTS) 연동 — 딥페이크 사기에 취약한 노년층의 골든타임 확보를 목표로 합니다.

### 9. 한계 및 주의 (Limitations)
* 제2축 음성은 진본 증명이 아니며, 양성 역시 절대적이지 않습니다(스푸핑 실증).
* 의미론 기반 탐지는 지름길 학습 위험이 있어 제1축과의 앙상블을 전제로만 동작합니다.
* 본 시스템은 **위험도 점수**를 제시하며 단정적 판정을 내리지 않습니다.
* `yt-dlp` 다운로드는 약관상 회색지대로, 본 프로젝트는 **학술 연구 목적**으로만 사용합니다.

---

## 📚 References (2024–2026, verified)

전체 검증 이력·상태 표기는 `docs/literature-review/` 참조.

**Surveys (per axis)**
* Xu et al., Fully AI-Generated Image Detection... (arXiv:2502.19716)
* Nguyen-Le et al., Passive Deepfake Detection... (arXiv:2411.17911)
* Cao et al., Secure and Robust Watermarking... (arXiv:2510.02384)
* Zhao et al., SoK: Watermarking for AI-Generated Content (arXiv:2411.18479)
* Hou et al., Detecting AI-Generated Video... (ACL Findings 2026)

**Axis 1 — Signal / Frequency**
* Frequency-Aware Robustness Analysis of Deepfake Detection Models (JAI, 2026.3)
* Hybrid Spatial–Frequency Attention with EfficientNet (Scientific Reports, 2026.4)
* Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking (ACM TOMMCCAP, 2026.3)
* Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale (ICLR 2026)
* Dual Data Alignment (DDA) (NeurIPS 2025)

**Axis 2 — Provenance**
* Zhao et al., Invisible Image Watermarks Are Provably Removable... (NeurIPS 2024)
* Müller & Debus, The Watermark Shortcut: How Provenance Marking Sabotages... (arXiv:2606.23335)
* Nemecek et al., Authenticated Contradictions from Desynchronized Provenance... (CVPR 2026 Workshops)
* Meta, Video Seal: Open and Efficient Video Watermarking (arXiv:2412.09492)

**Axis 3 & Fusion**
* Shuai et al., When Detectors Forget Forensics: Blocking Semantic Shortcuts (arXiv:2603.09242)
* Detecting AI-Generated Images via Distributional Deviations (arXiv:2601.03586)
* Physics-Driven Spatiotemporal Modeling (NeurIPS 2025)
* SpecSem-Net: Integrating Spectral and Semantic Features (arXiv:2605.17311)

**Benchmarks (Image & Video)**
* GenVidBench (arXiv:2501.11340) / DeMamba (Sci. China Inf. Sci. 2026) / RobustSora (arXiv:2512.10248)
* GenImage (NeurIPS 2023) / WildFake (AAAI 2025) / MIRAGE (AAAI 2026)

---
🎓 **Academic Research Project** — 창원대학교 컴퓨터공학과 학부 졸업 프로젝트. 본 시스템은 연구 및 교육 목적으로 확률적 위험 점수를 제공하며, 법적/재무적 결정의 유일한 증거로 사용될 수 없습니다.
