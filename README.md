# 🛡️ TRINITY v3: 3-Axis Ensemble AI-Generated Media Detector

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

> ⚠️ **v3 (2026-07) — 선행연구 검증 반영 / Research Prototype**
> 2024–2026 문헌 20여 편의 교차 검증(`docs/literature-review/` 참조) 결과를 설계에 반영했습니다: ① 제2축 **Early-Exit 정책 폐기·완화** (mark-to-frame 스푸핑 실증에 대응) ② **무결성 충돌 교차 감사(cross-layer audit) 모듈 신설** ③ **학습 데이터 위생(지름길 학습 3종 차단) 명문화** ④ 참고문헌을 검증 완료 문헌 기준으로 전면 갱신.
> v3 reflects a verified literature review: ① retirement of the axis-2 early-exit policy (countering proven mark-to-frame spoofing), ② a new cross-layer integrity audit module, ③ codified training-data hygiene against shortcut learning, ④ fully refreshed references.
> v3では検証済み文献レビューを反映: ①第2軸Early-Exitの廃止・緩和(mark-to-frameスプーフィング対応) ②整合性衝突クロス監査モジュール新設 ③学習データ衛生(ショートカット学習遮断)の明文化 ④参考文献の全面更新。

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
│   │   ├── synthid.py           # [KR] SynthID 플러그인 슬롯 / [EN] SynthID plugin slot / [JP] SynthIDプラグインスロット
│   │   └── audit.py             # [KR] 무결성 충돌 교차 감사 (v3 신설) / [EN] Cross-layer integrity audit (new in v3) / [JP] 整合性衝突クロス監査 (v3新設)
│   ├── 📂 axis_semantic/        # [KR] 3축: 의미·물리 / [EN] Axis 3: Semantic & Physics / [JP] 第3軸: 意味・物理
│   │   ├── distribution.py      # [KR] 분포 편차 + 의미 감산 / [EN] Distributional deviation + semantic subtraction / [JP] 分布偏差+意味減算
│   │   ├── nsg.py               # [KR] 시공간 그래디언트(NSG) / [EN] Spatiotemporal gradient / [JP] 時空間勾配
│   │   └── consistency.py       # [KR] 프레임 일관성(DeCoF) / [EN] Frame consistency / [JP] フレーム一貫性
│   ├── 📂 fusion/
│   │   └── specsem.py           # [KR] 게이트 융합(SpecSem-Net식) / [EN] Gated fusion / [JP] ゲート融合
│   └── ensemble.py              # [KR] 동적 가중 앙상블 / [EN] Dynamic weighted ensemble / [JP] 動的加重アンサンブル
│
├── 📂 preprocessing/
│   ├── ingest.py                # [KR] 투트랙 입력 어댑터(이미지/영상) / [EN] Two-track input adapter / [JP] 2トラック入力アダプタ
│   ├── youtube.py               # [KR] yt-dlp 다운로드 + 채널 ID 추출 / [EN] yt-dlp + channel ID / [JP] yt-dlp+チャンネルID抽出
│   ├── sampler.py               # [KR] 다중 프레임 샘플링(3-Point Biopsy) / [EN] Multi-frame sampling / [JP] 多フレームサンプリング
│   └── native_patch.py          # [KR] 원본 해상도 패치 추출 / [EN] Native-resolution patches / [JP] ネイティブ解像度パッチ
│
├── 📂 training/                 # [KR] 학습 파이프라인 + 데이터 위생 (v3) / [EN] Training pipeline + data hygiene (v3) / [JP] 学習パイプライン+データ衛生 (v3)
│   └── hygiene.py               # [KR] 압축 정렬·워터마크 탈상관·의미 감산 / [EN] Compression alignment, watermark decorrelation, semantic subtraction / [JP] 圧縮整列・透かし脱相関・意味減算
│
├── 📂 infrastructure/           # [KR] 설정·비동기·로깅 / [EN] Config, async, logging / [JP] 設定・非同期・ロギング
├── 📂 jobs/                     # [KR] 백그라운드 분석 작업 / [EN] Background analysis tasks / [JP] バックグラウンド分析タスク
├── 📂 storage/                  # [KR] 캐시·모델·채널 이력 DB / [EN] Cache, models, channel history DB / [JP] キャッシュ・モデル・履歴DB
├── 📂 deploy/cloudflare/        # [KR] 데모용 보안 터널 / [EN] Secure tunnel for demos / [JP] デモ用セキュアトンネル
├── 📂 docs/literature-review/   # [KR] 선행연구 검토(검증판) / [EN] Verified literature review / [JP] 検証済み文献レビュー
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

Sora·Veo 등 확산 모델 기반 영상 생성 기술은 인간의 시각적 인지 능력을 넘어서는 극사실적 결과물을 만들어내고 있으며, 위협 모델도 "얼굴 교체 딥페이크"에서 **"완전 생성 영상"**으로 이동했습니다. 최신 서베이(ACL Findings 2026)는 이 변화를 **"아티팩트 매칭 → 사실적 충실도 검증(Factual Fidelity Verification)"의 패러다임 전환**으로 정의합니다. 공간 픽셀 단서에만 의존하는 단일 모델 탐지기는 압축·노이즈·리사이즈 같은 통상적 유통 과정에서 단서를 잃고, 학습에 없던 생성기(Unseen Generators) 앞에서 일반화에 실패합니다. TRINITY는 **서로 실패 모드가 겹치지 않는 세 가지 단서**를 결합해 이 한계를 돌파합니다.

추가로, 분석 결과를 **유튜브 채널 ID 단위로 누적**하여 "이 채널에서 분석된 N개 영상 중 M개가 AI 생성 의심" 같은 채널 신뢰도 맥락을 함께 제공합니다.

### 2. 시스템 파이프라인 (Pipeline)

1. **Flutter 앱**: 유튜브 공유 시트로 링크 수신, 또는 갤러리에서 이미지 업로드 → REST API 호출
2. **제어**: FastAPI + Celery + Redis 큐 기반 비동기 처리 (요청 접수 → 백그라운드 분석 → 완료 통지)
3. **수집**: 영상 트랙은 `yt-dlp` 다운로드 + 채널 ID 추출, 이미지 트랙은 파일 직수신
4. **전처리**: 전/중/후 구간 다중 프레임 샘플링(3-Point Biopsy). **강제 리사이즈 금지** — 고주파 위조 흔적 보존을 위해 원본 해상도 패치를 유지하는 Native-Scale 원칙(ICLR 2026) 적용
5. **분석**: 3축 병렬 실행 → 무결성 충돌 교차 감사 → 게이트 융합 → 위험 점수 + 판단 근거 산출
6. **저장/반환**: 결과를 채널 이력 DB에 누적하고 앱에 반환

### 3. 3축 탐지 엔진 (The 3-Axis Engine)

#### 제1축 — 신호 (Signal / Frequency)
> *"생성 모델의 업샘플링 연산은 주파수 도메인에 주기적 스펙트럼 이상을 남긴다."*

- **원리**: 생성 모델의 목적 함수는 주파수 정보를 상대적으로 무시하도록 설계되어 있어, 진짜/생성물 간 불일치가 공간 도메인보다 주파수 도메인에서 증폭됩니다(귀납적 편향 관점, arXiv:2502.19716). 픽셀 도메인에서는 보이지 않는 이 흔적이 FFT/DCT로 투영하면 격자 피크·스펙트럼 이상으로 드러나며, 고대역 마스크 필터링 후 iFFT로 재구성한 **스펙트럼 잔차 맵**은 텍스처 이상과 주기적 그리드 아티팩트를 강하게 부각시킵니다.
- **구현**: ① 스펙트럼 잔차 추출, ② **EfficientNet + CBAM 공간-주파수 하이브리드 어텐션** — RGB 시각 표현과 DCT 주파수 요소를 조기 결합 (FaceForensics++ C23에서 ROC-AUC 0.997 보고), ③ **주파수 도메인 마스킹 학습** — 특정 대역 과적합을 방지하고 unseen generator 일반화를 강화, ④ FreqNet 계열의 위상+진폭 스펙트럼 합성곱 학습.
- **강건성 근거**: 2026년 강건성 정량화 연구에서 가우시안 블러가 JPEG 압축 대비 평균 30배 파괴적임이 확인됨 — 압축 위주인 유튜브 재인코딩 환경에서 주파수 단서는 상당 부분 생존합니다. 저화질(업스케일성 열화) 입력에는 품질 게이팅으로 가중치를 낮춥니다.

#### 제2축 — 출처 (Provenance)
> *"양성은 강한 증거, 음성은 무정보 — 그리고 양성조차 절대적이지 않다."*

| 기술 | 삽입 위치 | 성격 | 내구성 |
|---|---|---|---|
| **C2PA** | 파일 메타데이터 헤더(서명된 매니페스트) | 제작자·도구·편집 이력 등 정보 풍부 | 낮음 — 재인코딩·스크린샷·플랫폼 업로드로 유실 |
| **SynthID류** | 콘텐츠 자체(픽셀/스펙트로그램) | AI 생성 여부만 표시(정보 빈약) | 높음 — 압축·크롭·리사이즈·색보정 생존 |

- **커버리지**: 2026년 5월 기준 OpenAI(ChatGPT·API 이미지)가 SynthID를 채택하고 C2PA 운영위에 합류하는 등 업계 표준화가 진행 중이며(누적 100억 개 이상 워터마킹, 2025년 공식 발표 기준), EU AI Act 제50조(2026.8 발효)와 캘리포니아 SB 942(2026.1 발효)가 합성 콘텐츠의 기계판독형 표시를 법제화하여 커버리지는 계속 확대될 전망입니다.
- **판정 정책 (v3 개정)**: 워터마크/매니페스트 신호는 **긍정 증거(positive evidence)로만** 사용합니다. 음성은 "정보 없음"입니다 — 다수 생성기가 워터마크를 넣지 않으며, 확산 정제(diffusion purification) 재생성 공격으로 상용 워터마크도 품질 저하 없이 세탁될 수 있음이 증명되었습니다(NeurIPS 2024). **v2의 "양성 → 즉시 확정(Early-Exit)" 정책은 폐기합니다** — 진짜 미디어에 워터마크를 덧칠해 가짜로 몰아가는 스푸핑(mark-to-frame) 공격이 실증되었기 때문입니다(탐지기 EER 16%→75% 급등, arXiv:2606.23335). 양성 시에는 고신뢰 가중치를 부여하되, 제1·3축의 경량 교차 확인을 반드시 병행합니다.
- **무결성 충돌 교차 감사 (v3 신설, `audit.py`)**: C2PA 검증(메타데이터 서명 확인)과 워터마크 검출(픽셀 신호 디코딩)은 서로를 참조하지 않는 독립 절차라서, "사람이 편집함"을 주장하는 유효한 C2PA 매니페스트와 "AI 생성"을 알리는 픽셀 워터마크가 한 파일에 공존할 수 있습니다 — 이른바 **"인증된 가짜(Authenticated Fakes)"**(CVPR 2026 Workshops). TRINITY는 두 계층의 결과를 충돌 매트릭스로 교차 감사하여, 유효 매니페스트의 주장과 워터마크 신호가 모순되면 **세탁 의심 신호로 승격**합니다. 원 논문의 교차 감사 프로토콜은 이미지에서 100% 분류 정확도를 보고했으며, **영상 모달리티 확장은 미해결 과제로 남아 있어 본 프로젝트의 기여 지점**입니다.
- **구현 현실**: SynthID 검출은 구글 비공개 인프라가 필요하므로(제3자 독립 구현 불가), Content Detection API 얼리 액세스를 신청하고 승인 전까지는 **인터페이스만 정의된 플러그인 슬롯**으로 유지합니다. C2PA 파싱은 오픈 표준으로 직접 구현합니다.

#### 제3축 — 의미·물리 (Semantic & Physics)
> *"픽셀은 완벽해도, 의미의 분포와 물리 법칙은 속이기 어렵다."*

- **분포 편차 + 의미 감산**: 동결된 CLIP-ViT 특징 공간에서 실제 이미지 분포로부터의 편차를 측정하되, 모델이 피사체의 의미를 판별 기준으로 암기하는 **의미론적 지름길**을 수학적으로 차단합니다 — 마스킹 미세튜닝(MPFT/TAM, GenImage 98.2% 보고) 또는 SVD로 의미 우세 부분공간을 추정·감산하는 GSD-CLIP 계열(arXiv:2603.09242) 중 구현 단계에서 택일합니다.
- **물리 기반 시공간 모델링 (영상 전용)**: NSG(정규화 시공간 그래디언트)로 공간 확률 그래디언트와 시간 밀도 변화의 비율을 정량화하고, 실제 영상과의 MMD를 탐지 지표로 사용해 중력·반사·조명 등 **물리 법칙 위반**을 포착합니다 (NeurIPS 2025).
- **프레임 일관성 (영상 전용)**: DeCoF 계열 접근으로 인접 프레임 간 의미 정체성의 미세한 흔들림을 탐지합니다.
- **스코프 주석**: 최신 서베이의 4계층 분류(내재 단서 → 시공간 일관성 → 크로스모달 → LLM 세계 추론) 중 본 프로젝트는 1~2계층 + 분포 편차를 구현하며, 3~4계층(오디오-비디오 크로스모달, LLM 에이전트 추론)은 Future Work로 명시합니다.

#### 융합 — SpecSem-Net식 게이트 결합
제1축의 고주파 스펙트럼 특징은 배경의 정상적 환경 노이즈까지 포함해 오탐의 원인이 됩니다. TRINITY는 SpecSem-Net의 **게이트 융합 메커니즘**을 차용해, 제3축의 의미론적 컨텍스트로 스펙트럼 특징을 적응적으로 변조 — 무의미한 노이즈를 걸러내고 정제된 특징만 병합합니다. 해당 구조는 Sora·Veo급 생성물 상대로 87% 이상의 정확도를 보고했습니다.

**앙상블 정책 (v3)**:

| 조건 | 동작 |
|---|---|
| 제2축 양성 (워터마크/매니페스트 확인) | 고신뢰 가중 + 제1·3축 경량 교차 확인 (스푸핑 대비 — 즉시 확정하지 않음) |
| 무결성 충돌 감지 (매니페스트 주장 ↔ 워터마크 모순) | 세탁 의심 신호로 승격, 전 축 정밀 분석 |
| 제2축 음성 | 제1축 + 제3축 게이트 융합이 주력 |
| 저화질/블러성 열화 입력 | 제1축 가중 하향, 제3축 가중 상향 (품질 게이팅) |
| 얼굴 유무 | 무관 — 전 축이 인물 없는 완전 생성 영상에도 동작 |

### 4. 투트랙: 이미지 + 영상 (Two-Track)

세 축 모두 본질적으로 프레임 단위 분석이므로, 코어 모듈을 100% 공유하고 입력 어댑터만 분기합니다.

| 축 | 이미지 트랙 | 영상 트랙 |
|---|---|---|
| 1축 신호 | ✅ (압축 적어 신호 선명) | ✅ 다중 프레임 집계 + 원본 해상도 패치 |
| 2축 출처 | ✅ C2PA + 워터마크 + 교차 감사 | ⚠️ 재인코딩 시 C2PA 유실 가능, 워터마크는 생존 — 영상 교차 감사는 본 프로젝트가 개척 |
| 3축 의미 | ✅ 분포 편차 + 의미 감산 | ✅ 분포 편차 + NSG + 프레임 일관성 |

### 5. 벤치마크 검증 체계 (Benchmarks)

| 벤치마크 | 규모/특징 | 검증 목적 |
|---|---|---|
| **GenVidBench** (2025.1) | Sora 포함 11개 SOTA 생성기, 14.3만 영상, 교차 소스·교차 생성기 | 일반화 성능의 기준점 |
| **GenVideo / DeMamba** (2024.5) | 100만+ 생성 영상, T2V·I2V 포괄 | 유통 압축·화질 열화 저항성 |
| **RobustSora** (2025.12) | 워터마크 제거(De-watermarked) 영상 중심 | 워터마크 회피 공격 하 제1·3축 독립 성능 분리 검증 |

**일반화 → 압축 저항 → 워터마크 회피 공격**으로 이어지는 3단 스트레스 테스트 체계입니다.

### 6. 학습 데이터 위생 — 지름길 학습 차단 (v3 신설)

2024~2026 문헌은 세 축 각각에서 같은 병리의 변종을 독립적으로 보고했습니다: **탐지기는 항상 가장 값싼 상관관계를 암기한다.** TRINITY는 아래 차단책을 학습 파이프라인의 요구사항(`training/hygiene.py`)으로 명문화합니다.

| 축 | 지름길 | 발생 원인 | 차단책 (근거) |
|---|---|---|---|
| 1축 신호 | "고주파 풍부 = 가짜" | 진짜(웹 재압축) vs 가짜(원시 생성)의 압축 비대칭 | DDA식 압축·스펙트럼 정렬 (NeurIPS 2025) |
| 2축 출처 | "워터마크 = 가짜" | 합성물에만 워터마크가 존재하는 훈련 데이터 | 양 클래스 워터마크 탈상관 — 제거본 혼합 또는 진짜에도 부여 (arXiv:2606.23335, WASP) |
| 3축 의미 | "특정 피사체 = 가짜" | 생성 데이터셋의 의미 분포 쏠림 | 의미 부분공간 감산(SVD) 또는 마스킹 미세튜닝 (arXiv:2603.09242) |

훈련 전 세 항목을 점검하고, 평가는 5장의 3단 벤치마크로 스트레스 테스트합니다.

### 7. 서버 하드웨어 전략 (Apple Silicon Edge Server)

분석 서버는 Apple Silicon 맥북(현재 MacBook Air 기반 개발)에서 구동되며, 연산 특성별로 이기종 코어를 분할 사용합니다.

- **FFT/주파수 변환 → CPU(AMX)·커스텀 Metal 커널**: PyTorch `torch.fft`는 MPS 백엔드에서 2의 거듭제곱 텐서 제약과 복소수 처리 이슈가 보고되어, 주파수 연산은 CPU 행렬 보조프로세서 경로로 우회합니다.
- **출처 스캐닝·교차 감사 → 효율 코어**: C2PA 파싱·해시 검증·충돌 매트릭스 판정은 암호학적/논리 연산이므로 효율 코어 백그라운드 스레드에서 파일 로딩과 동시 수행합니다.
- **ViT/CNN 추론 → CoreML(ANE)**: 대형 비전 모델은 FP16 정적 양자화 후 Neural Engine에 할당 — MPS 대비 약 1.5배 속도 향상이 보고된 경로입니다.
- **열 관리**: 팬리스 구조의 열 스로틀링을 감안한 분할 실행으로 안정적 처리량을 유지하고, 데모 시에는 Cloudflare Tunnel로 외부 접근을 제공합니다.

### 8. Flutter 클라이언트 & 접근성 (Client & Accessibility)

- **진입**: OS 공유 시트("TRINITY로 분석") 또는 앱 내 링크 붙여넣기/이미지 업로드
- **화면 흐름**: 홈 → 분석 중 → 결과(점수+근거+채널 요약) → 채널 상세 — 탭바 없는 스택 네비게이션
- **디자인**: Bold Typography 다크 테마 (배경 `#0A0A0A`, 버밀리언 액센트 `#FF3D00`, Inter Tight + JetBrains Mono) — 판독 점수를 화면을 지배하는 헤드라인으로 제시
- **신호등 메타포**: 초록(낮은 위험) / 노랑(주의 — 출처 미확인·무결성 충돌 등) / 빨강(높은 위험)의 3단계 위험도 + 축별 근거 칩
- **취약 계층 배려**: 고대비 팔레트, 동적 글꼴 크기, 화면 낭독기(VoiceOver/TalkBack) 연동, 판독 결과의 TTS 음성 안내 — 딥페이크 금융 사기의 주요 표적인 노년층의 골든타임 확보를 목표로 합니다

### 9. 한계 및 주의 (Limitations)

- 제2축 **음성은 진본 증명이 아니며**(워터마크 미적용 생성기 다수, 악의적 제거 가능), **양성 역시 절대적이지 않습니다**(mark-to-frame 스푸핑 실증) — 교차 감사와 타 축 병행으로 완화하되, 잔여 위험을 명시합니다.
- 의미론 기반 탐지는 지름길 학습 위험이 보고되어 있어 **제1축과의 앙상블 + 6장의 위생 절차를 전제**로만 동작합니다.
- NSG 등 일부 모듈은 연산 비용이 높아 품질·길이 조건에 따라 게이팅됩니다.
- 크로스모달·LLM 세계지식 추론 계층은 범위 외이며 Future Work로 명시합니다.
- 본 시스템은 **위험도 점수**를 제시하며 단정적 판정을 내리지 않습니다.
- `yt-dlp` 기반 다운로드는 플랫폼 약관상 회색지대로, 본 프로젝트는 **학술 연구 목적**으로만 사용합니다.

---

<br>
## 🇺🇸 English

### 1. Overview

**TRINITY** is a **3-axis ensemble detection system** that determines whether media shared via a YouTube link (Share Sheet) or an uploaded image was **generated or manipulated by AI**. Users interact through a lightweight Flutter cross-platform app, while analysis runs on a **self-hosted edge server** (FastAPI) on an Apple Silicon MacBook — media never leaves for a commercial cloud, minimizing privacy exposure.

Diffusion-based generators such as Sora and Veo now produce hyper-realistic footage beyond human perceptual limits, and the threat model has shifted from face-swap deepfakes to **fully generated video**. The latest survey of the field (ACL Findings 2026) frames this as a paradigm shift **from artifact matching to Factual Fidelity Verification**. Single-model detectors that rely on spatial pixel cues lose their evidence under routine distribution transforms and fail to generalize to unseen generators. TRINITY overcomes this by combining **three cues with non-overlapping failure modes**.

Results are also **aggregated per YouTube channel ID**, providing context such as "M of N analyzed videos from this channel were flagged as likely AI-generated."

### 2. System Pipeline

1. **Flutter app**: receives a YouTube link via the OS share sheet, or an image upload → calls the REST API
2. **Control**: FastAPI + Celery + Redis queue-based async processing
3. **Ingestion**: video track via `yt-dlp` download + channel ID extraction; image track via direct file upload
4. **Preprocessing**: multi-frame sampling across start/middle/end segments (3-Point Biopsy). **No forced resizing** — native-resolution patches preserve high-frequency forgery traces (Native-Scale principle, ICLR 2026)
5. **Analysis**: three axes in parallel → cross-layer integrity audit → gated fusion → risk score + evidence
6. **Store/return**: results accumulate in the channel-history DB and return to the app

### 3. The 3-Axis Detection Engine

#### Axis 1 — Signal (Frequency)
> *"Generative up-sampling leaves periodic spectral anomalies in the frequency domain."*

- **Principle**: generative training objectives comparatively neglect frequency information, so real-vs-generated discrepancies are amplified in the frequency domain relative to pixel space (inductive-priors view, arXiv:2502.19716). Invisible in pixels, these traces emerge as grid peaks and spectral anomalies under FFT/DCT; a **spectral residual map** (high-pass masking + iFFT) strongly amplifies texture anomalies and periodic grid artifacts.
- **Implementation**: ① spectral residual extraction; ② **hybrid spatial-frequency attention (EfficientNet + CBAM)** fusing RGB with DCT components early (ROC-AUC 0.997 reported on FaceForensics++ C23); ③ **frequency-domain masking** during training against single-band over-fitting; ④ FreqNet-style convolution on phase and amplitude spectra.
- **Robustness evidence**: a 2026 quantitative study found Gaussian blur roughly 30× more destructive than JPEG compression — frequency cues largely survive YouTube's compression-centric re-encoding. Low-quality inputs are down-weighted via quality gating.

#### Axis 2 — Provenance
> *"A positive is strong evidence; a negative is no information — and even a positive is not absolute."*

| Technology | Embedded in | Character | Durability |
|---|---|---|---|
| **C2PA** | file metadata header (signed manifest) | information-rich: creator, tool, edit history | Low — lost to re-encoding, screenshots, platform uploads |
| **SynthID-class** | the content itself (pixels/spectrogram) | information-poor: AI-origin signal only | High — survives compression, cropping, resizing, color edits |

- **Coverage**: as of May 2026, OpenAI adopted SynthID for ChatGPT/API images and joined the C2PA steering committee (10B+ items watermarked per official 2025 figures), while the EU AI Act Article 50 (in force Aug 2026) and California SB 942 (Jan 2026) legally mandate machine-readable marking of synthetic content — coverage will keep expanding.
- **Verdict policy (revised in v3)**: watermark/manifest signals are used as **positive evidence only**. A negative means "no information" — many generators embed nothing, and diffusion-purification regeneration attacks provably launder commercial watermarks without visible quality loss (NeurIPS 2024). **The v2 "positive → immediate confirmation (early exit)" policy is retired**: mark-to-frame spoofing — overlaying a watermark onto genuine media to frame it as fake — has been empirically demonstrated (detector EER lifted from 16% to 75%, arXiv:2606.23335). A positive now yields a high-confidence weight, always accompanied by lightweight cross-checks from Axes 1 & 3.
- **Cross-layer integrity audit (new in v3, `audit.py`)**: C2PA validation (a cryptographic signature check over metadata) and watermark detection (a signal-decoding operation over pixels) never condition on each other — so a file can carry a valid C2PA manifest asserting "human-edited" while its pixels carry an "AI-generated" watermark: **Authenticated Fakes** (CVPR 2026 Workshops). TRINITY cross-audits both layers through a conflict matrix and **escalates contradictions as laundering-suspicion signals**. The original audit protocol reported 100% classification accuracy on images; **extending it to the video modality remains open — a contribution point of this project**.
- **Implementation reality**: SynthID detection requires Google's proprietary infrastructure, so TRINITY registers for the Content Detection API early access and keeps the module as an **interface-defined plugin slot** until approved. C2PA parsing is implemented directly as an open standard.

#### Axis 3 — Semantic & Physics
> *"Pixels can be perfect; semantic distributions and physical laws are harder to fake."*

- **Distributional deviation + semantic subtraction**: measures deviation from the real-image distribution in a frozen CLIP-ViT feature space while mathematically blocking the **semantic shortcut** (memorizing subject identity as the fake criterion) — choosing at implementation time between masking-based fine-tuning (MPFT/TAM, 98.2% reported on GenImage) and GSD-CLIP-style SVD subtraction of the semantically dominant subspace (arXiv:2603.09242).
- **Physics-driven spatiotemporal modeling (video only)**: NSG quantifies the ratio of spatial probability gradients to temporal density changes, using MMD against real-video statistics to catch **violations of physical dynamics** (NeurIPS 2025).
- **Frame consistency (video only)**: DeCoF-style detection of subtle identity jitter between adjacent frames.
- **Scope note**: of the four-layer taxonomy in the latest survey (intrinsic cues → spatiotemporal consistency → cross-modal → LLM world reasoning), this project implements layers 1–2 plus distributional deviation; layers 3–4 are explicitly Future Work.

#### Fusion — SpecSem-Net-style gated merging
Semantic context from Axis 3 adaptively modulates the spectral features from Axis 1, filtering benign environmental noise before merging — feature-level fusion, not naive score voting (87%+ accuracy reported against Sora/Veo-class generators).

**Ensemble policy (v3)**:

| Condition | Behavior |
|---|---|
| Axis 2 positive (watermark/manifest verified) | High-confidence weight + lightweight cross-check by Axes 1 & 3 (anti-spoofing — no immediate confirmation) |
| Integrity clash detected (manifest claim ↔ watermark contradiction) | Escalate as laundering suspicion; full-axis deep analysis |
| Axis 2 negative | Gated fusion of Axes 1 + 3 takes over |
| Low-quality / blur-degraded input | Axis 1 down-weighted, Axis 3 up-weighted (quality gating) |
| Face presence | Irrelevant — all axes work on fully generated, faceless footage |

### 4. Two-Track: Image + Video

| Axis | Image track | Video track |
|---|---|---|
| 1 Signal | ✅ (cleaner signal, less compression) | ✅ multi-frame aggregation + native-resolution patches |
| 2 Provenance | ✅ C2PA + watermark + cross-audit | ⚠️ C2PA may be stripped by re-encoding; watermarks survive — video cross-audit pioneered by this project |
| 3 Semantic | ✅ deviation + semantic subtraction | ✅ deviation + NSG + frame consistency |

### 5. Benchmark Validation

| Benchmark | Scale / Features | Purpose |
|---|---|---|
| **GenVidBench** (2025.1) | 11 SOTA generators incl. Sora, 143K videos, cross-source & cross-generator | Baseline for generalization |
| **GenVideo / DeMamba** (2024.5) | 1M+ generated videos, T2V & I2V | Resistance to distribution compression/degradation |
| **RobustSora** (2025.12) | De-watermarked videos | Isolated validation of Axes 1 & 3 under watermark-removal attacks |

A three-stage stress regime: **generalization → compression resistance → watermark-evasion attack**.

### 6. Training Data Hygiene — Blocking Shortcut Learning (new in v3)

The 2024–2026 literature independently reports the same pathology in each axis: **detectors always memorize the cheapest correlation.** TRINITY codifies the countermeasures as training-pipeline requirements (`training/hygiene.py`).

| Axis | Shortcut | Cause | Countermeasure (basis) |
|---|---|---|---|
| 1 Signal | "rich high-freq = fake" | compression asymmetry: real (web-recompressed) vs fake (pristine) | DDA-style compression/spectrum alignment (NeurIPS 2025) |
| 2 Provenance | "watermark = fake" | watermarks present only on synthetic training samples | decorrelate watermarks across both classes — mix stripped fakes or watermark reals (arXiv:2606.23335, WASP) |
| 3 Semantic | "certain subjects = fake" | semantic skew of generated datasets | SVD subtraction of semantic subspace or masked fine-tuning (arXiv:2603.09242) |

All three are checked before training; evaluation stress-tests via the three-stage benchmarks in §5.

### 7. Server Hardware Strategy (Apple Silicon Edge Server)

- **FFT/frequency transforms → CPU (AMX) / custom Metal kernels**: PyTorch `torch.fft` on MPS has power-of-two constraints and complex-type issues, so frequency ops route through the CPU matrix coprocessor path.
- **Provenance scanning & audit → efficiency cores**: C2PA parsing, hash verification, and conflict-matrix logic are cryptographic/logical, running on efficiency-core background threads concurrently with file loading.
- **ViT/CNN inference → CoreML (ANE)**: large vision models statically quantized to FP16 on the Neural Engine (~1.5× over MPS reported).
- **Thermals**: split execution accounts for fanless throttling; demos exposed via Cloudflare Tunnel.

### 8. Flutter Client & Accessibility

- **Entry**: OS share sheet ("Analyze with TRINITY") or in-app link paste / image upload
- **Flow**: Home → Analyzing → Result (score + evidence + channel summary) → Channel Detail — stack navigation, no tab bar
- **Design**: Bold Typography dark theme (background `#0A0A0A`, vermillion accent `#FF3D00`, Inter Tight + JetBrains Mono) — the verdict score as a screen-dominating headline
- **Traffic-light metaphor**: green (low risk) / yellow (caution — e.g., provenance unverified or integrity clash) / red (high risk), with per-axis evidence chips
- **Accessibility**: high-contrast palette, dynamic font scaling, screen-reader integration, and TTS verdicts — protecting older adults, the primary targets of deepfake financial fraud

### 9. Limitations

- An Axis-2 **negative is not proof of authenticity**, and **a positive is not absolute either** (mark-to-frame spoofing is demonstrated) — mitigated by the cross-layer audit and multi-axis checks, with residual risk stated explicitly.
- Semantic detection carries documented shortcut-learning risks and operates **only in ensemble with Axis 1 plus the hygiene procedures of §6**.
- Compute-heavy modules (e.g., NSG) are gated by input quality and length.
- Cross-modal and LLM world-reasoning layers are out of scope (Future Work).
- The system reports a **risk score** and never issues a categorical verdict.
- `yt-dlp` downloads sit in a platform-ToS gray area; this project is **for academic research only**.

---

<br>
## 🇯🇵 日本語 (Japanese)

### 1. 概要 (Overview)

**TRINITY**は、YouTubeリンクの共有(Share Sheet)または画像アップロードで届いたメディアが**AIによって生成・操作されたか**を判別する**3軸アンサンブル検知システム**です。ユーザーは軽量なFlutterアプリで操作し、分析はApple Silicon MacBook上の**セルフホスト型エッジサーバー**(FastAPI)が実行します。メディアを商用クラウドへ送信しない構成でプライバシー露出を最小化します。

SoraやVeoに代表される拡散モデル系の映像生成は人間の知覚限界を超える超写実的な結果を生み、脅威モデルは「顔交換ディープフェイク」から**「完全生成映像」**へ移行しました。最新サーベイ(ACL Findings 2026)はこの変化を**「アーティファクトマッチング → 事実的忠実度検証(Factual Fidelity Verification)」のパラダイム転換**と定義しています。TRINITYは**互いに失敗モードが重ならない3つの手掛かり**を結合してこの限界を突破します。

さらに分析結果を**YouTubeチャンネルID単位で蓄積**し、「このチャンネルで分析されたN本中M本がAI生成の疑い」といったチャンネル信頼度の文脈も提供します。

### 2. システムパイプライン (Pipeline)

1. **Flutterアプリ**: 共有シートからYouTubeリンク受信、または画像アップロード → REST API呼び出し
2. **制御**: FastAPI + Celery + Redis キューによる非同期処理
3. **取得**: 映像トラックは`yt-dlp`ダウンロード+チャンネルID抽出、画像トラックはファイル直接受信
4. **前処理**: 冒頭・中盤・終盤の多フレームサンプリング(3-Point Biopsy)。**強制リサイズ禁止** — ネイティブ解像度パッチで高周波の偽造痕跡を保存(Native-Scale原則、ICLR 2026)
5. **分析**: 3軸並列実行 → 整合性衝突クロス監査 → ゲート融合 → リスクスコア+判断根拠
6. **保存/返却**: 結果をチャンネル履歴DBへ蓄積しアプリへ返却

### 3. 3軸検知エンジン (The 3-Axis Engine)

#### 第1軸 — 信号 (Signal / Frequency)
> *「生成モデルのアップサンプリング演算は、周波数ドメインに周期的なスペクトル異常を残す。」*

- **原理**: 生成モデルの目的関数は周波数情報を相対的に無視するよう設計されており、実物/生成物間の不一致はピクセル空間より周波数ドメインで増幅されます(帰納的バイアスの観点、arXiv:2502.19716)。FFT/DCTで投影すると格子状ピークやスペクトル異常として現れ、高域マスク+iFFTで再構成した**スペクトル残差マップ**がテクスチャ異常と周期的グリッドを強く浮かび上がらせます。
- **実装**: ① スペクトル残差抽出、② **EfficientNet + CBAM 空間-周波数ハイブリッドアテンション**(FaceForensics++ C23でROC-AUC 0.997の報告)、③ **周波数ドメインマスキング学習**で特定帯域への過学習を防止、④ FreqNet系の位相+振幅スペクトル畳み込み学習。
- **頑健性の根拠**: 2026年の定量研究でガウシアンブラーがJPEG圧縮の約30倍破壊的と確認 — 圧縮中心のYouTube再エンコード環境で周波数の手掛かりは相当程度生存します。低画質入力は品質ゲーティングで重みを下げます。

#### 第2軸 — 出所 (Provenance)
> *「陽性は強い証拠、陰性は無情報 — そして陽性さえ絶対ではない。」*

| 技術 | 埋め込み位置 | 性格 | 耐久性 |
|---|---|---|---|
| **C2PA** | ファイルメタデータヘッダ(署名付きマニフェスト) | 制作者・ツール・編集履歴など情報豊富 | 低 — 再エンコード・スクリーンショット・投稿で消失 |
| **SynthID系** | コンテンツ自体(ピクセル/スペクトログラム) | AI生成の有無のみ(情報は乏しい) | 高 — 圧縮・クロップ・リサイズ・色調整で生存 |

- **カバレッジ**: 2026年5月時点でOpenAIがChatGPT/API画像へSynthIDを採用しC2PA運営委員会に参加(累計100億件以上、2025年公式発表基準)。EU AI法第50条(2026年8月施行)と加州SB 942(2026年1月施行)が機械判読型表示を法制化し、カバレッジは拡大見込みです。
- **判定ポリシー (v3改訂)**: 透かし/マニフェスト信号は**肯定的証拠(positive evidence)としてのみ**使用します。陰性は「情報なし」です — 多くの生成器は透かしを埋め込まず、拡散精製(diffusion purification)による再生成攻撃で商用透かしも品質低下なしに洗浄できることが証明されています(NeurIPS 2024)。**v2の「陽性 → 即時確定(Early-Exit)」ポリシーは廃止します** — 本物のメディアに透かしを上塗りして偽物に仕立てるスプーフィング(mark-to-frame)攻撃が実証されたためです(検知器EER 16%→75%、arXiv:2606.23335)。陽性時は高信頼の重みを与えつつ、第1・3軸の軽量クロスチェックを必ず併用します。
- **整合性衝突クロス監査 (v3新設、`audit.py`)**: C2PA検証(メタデータ署名確認)と透かし検出(ピクセル信号復号)は互いを参照しない独立手続きのため、「人間が編集」と主張する有効なC2PAマニフェストと「AI生成」を示すピクセル透かしが同一ファイルに共存し得ます — いわゆる**「認証された偽物(Authenticated Fakes)」**(CVPR 2026 Workshops)。TRINITYは両層の結果を衝突マトリクスでクロス監査し、矛盾を**洗浄疑い信号へ昇格**させます。原論文の監査プロトコルは画像で100%の分類精度を報告しており、**映像モダリティへの拡張は未解決課題 — 本プロジェクトの貢献点**です。
- **実装の現実**: SynthID検出はGoogleの非公開インフラが必要なため、Content Detection APIのアーリーアクセスを申請し、承認まで**インターフェース定義済みプラグインスロット**として保持。C2PA解析はオープン標準として直接実装します。

#### 第3軸 — 意味・物理 (Semantic & Physics)
> *「ピクセルは完璧でも、意味の分布と物理法則を偽るのは難しい。」*

- **分布偏差 + 意味減算**: 凍結CLIP-ViT特徴空間で実画像分布からの偏差を測定しつつ、被写体の意味を判別基準として暗記する**意味的ショートカット**を数学的に遮断します — マスキング微調整(MPFT/TAM、GenImage 98.2%報告)またはSVDで意味優勢部分空間を推定・減算するGSD-CLIP系(arXiv:2603.09242)から実装段階で択一します。
- **物理ベース時空間モデリング(映像のみ)**: NSGで空間確率勾配と時間密度変化の比率を定量化し、実映像統計とのMMDで**物理法則違反**を捕捉(NeurIPS 2025)。
- **フレーム一貫性(映像のみ)**: DeCoF系で隣接フレーム間の意味的アイデンティティの揺らぎを検知。
- **スコープ注記**: 最新サーベイの4階層(内在手掛かり → 時空間一貫性 → クロスモーダル → LLM世界推論)のうち本プロジェクトは1~2階層+分布偏差を実装し、3~4階層はFuture Workと明記します。

#### 融合 — SpecSem-Net式ゲート結合
第3軸の意味的コンテキストで第1軸のスペクトル特徴を適応的に変調し、良性の環境ノイズを濾過してから併合します — スコア投票ではなく特徴レベルの融合です(Sora・Veo級に対し87%以上の精度を報告)。

**アンサンブルポリシー (v3)**:

| 条件 | 動作 |
|---|---|
| 第2軸 陽性(透かし/マニフェスト確認) | 高信頼の重み + 第1・3軸の軽量クロスチェック(スプーフィング対策 — 即時確定しない) |
| 整合性衝突検知(マニフェスト主張 ↔ 透かし矛盾) | 洗浄疑い信号へ昇格、全軸精密分析 |
| 第2軸 陰性 | 第1軸+第3軸のゲート融合が主力 |
| 低画質/ブラー性劣化入力 | 第1軸の重みを下げ、第3軸を上げる(品質ゲーティング) |
| 顔の有無 | 無関係 — 全軸が人物のいない完全生成映像にも動作 |

### 4. 2トラック: 画像 + 映像 (Two-Track)

| 軸 | 画像トラック | 映像トラック |
|---|---|---|
| 第1軸 信号 | ✅(圧縮が少なく信号鮮明) | ✅ 多フレーム集計+ネイティブ解像度パッチ |
| 第2軸 出所 | ✅ C2PA+透かし+クロス監査 | ⚠️ 再エンコードでC2PA消失の可能性、透かしは生存 — 映像クロス監査は本プロジェクトが開拓 |
| 第3軸 意味 | ✅ 分布偏差+意味減算 | ✅ 分布偏差+NSG+フレーム一貫性 |

### 5. ベンチマーク検証体系 (Benchmarks)

| ベンチマーク | 規模/特徴 | 検証目的 |
|---|---|---|
| **GenVidBench** (2025.1) | Sora含む11のSOTA生成器、14.3万本、クロスソース・クロス生成器 | 一般化性能の基準点 |
| **GenVideo / DeMamba** (2024.5) | 100万本以上、T2V・I2V包括 | 流通圧縮・画質劣化への耐性 |
| **RobustSora** (2025.12) | 透かし除去(De-watermarked)映像中心 | 透かし回避攻撃下での第1・3軸の独立性能検証 |

**一般化 → 圧縮耐性 → 透かし回避攻撃**の3段階ストレステスト体系です。

### 6. 学習データ衛生 — ショートカット学習の遮断 (v3新設)

2024~2026年の文献は3軸それぞれで同じ病理の変種を独立に報告しています: **検知器は常に最も安価な相関を暗記する。** TRINITYは以下の遮断策を学習パイプラインの要件(`training/hygiene.py`)として明文化します。

| 軸 | ショートカット | 原因 | 遮断策(根拠) |
|---|---|---|---|
| 第1軸 信号 | 「高周波が豊富=偽物」 | 実物(Web再圧縮) vs 偽物(未圧縮生成)の非対称 | DDA式圧縮・スペクトル整列 (NeurIPS 2025) |
| 第2軸 出所 | 「透かし=偽物」 | 合成物のみに透かしがある学習データ | 両クラスで透かしを脱相関 — 除去版混合または実物にも付与 (arXiv:2606.23335, WASP) |
| 第3軸 意味 | 「特定の被写体=偽物」 | 生成データセットの意味分布の偏り | SVD意味部分空間減算またはマスキング微調整 (arXiv:2603.09242) |

学習前に3項目を点検し、評価は§5の3段ベンチマークでストレステストします。

### 7. サーバーハードウェア戦略 (Apple Silicon Edge Server)

- **FFT/周波数変換 → CPU(AMX)・カスタムMetalカーネル**: `torch.fft`のMPS制約(2のべき乗・複素数問題)を迂回。
- **出所スキャン・監査 → 効率コア**: C2PA解析・ハッシュ検証・衝突マトリクス判定は暗号学的/論理演算のため効率コアで並行実行。
- **ViT/CNN推論 → CoreML(ANE)**: FP16静的量子化でNeural Engineに割り当て(MPS比約1.5倍の報告)。
- **熱管理**: ファンレスのサーマルスロットリングを考慮した分割実行。デモはCloudflare Tunnelで公開。

### 8. Flutterクライアント & アクセシビリティ (Client & Accessibility)

- **入口**: OS共有シート(「TRINITYで分析」)またはアプリ内リンク貼り付け/画像アップロード
- **画面フロー**: ホーム → 分析中 → 結果 → チャンネル詳細 — タブバーなしのスタックナビゲーション
- **デザイン**: Bold Typographyダークテーマ(背景 `#0A0A0A`、バーミリオン `#FF3D00`、Inter Tight + JetBrains Mono)
- **信号機メタファー**: 緑(低リスク)/ 黄(注意 — 出所未確認・整合性衝突など)/ 赤(高リスク)+ 軸別根拠チップ
- **デジタル弱者への配慮**: 高コントラスト、動的フォント、スクリーンリーダー連携、TTS音声案内 — 高齢者のゴールデンタイム確保を目標

### 9. 制限と注意 (Limitations)

- 第2軸の**陰性は真正の証明ではなく**、**陽性も絶対ではありません**(mark-to-frameスプーフィング実証済み) — クロス監査と多軸併用で緩和しつつ、残余リスクを明記します。
- 意味ベース検知はショートカット学習リスクがあるため、**第1軸とのアンサンブル+§6の衛生手続きを前提**としてのみ動作します。
- NSGなど一部モジュールは演算コストが高く、品質・長さ条件でゲーティングされます。
- クロスモーダル・LLM世界推論の階層は範囲外(Future Work)です。
- 本システムは**リスクスコア**を提示し、断定的判定は行いません。
- `yt-dlp`ダウンロードは規約上グレーゾーンであり、本プロジェクトは**学術研究目的**に限定します。

---

## 📚 References (2024–2026, verified)

> 전체 검증 이력·상태 표기는 `docs/literature-review/`의 선행연구 검토 보고서를 참조. All entries below were existence-verified during the v3 review.

### Surveys (per axis)
- **Axis 1** — Xu et al., *Fully AI-Generated Image Detection: Definition, Recent Advances and Challenges* — [arXiv:2502.19716](https://arxiv.org/abs/2502.19716)
- **Axis 1** — Nguyen-Le et al., *Passive Deepfake Detection: A Comprehensive Survey across Multi-modalities* — [arXiv:2411.17911](https://arxiv.org/abs/2411.17911)
- **Axis 2** — Cao et al., *Secure and Robust Watermarking for AI-generated Images: A Comprehensive Survey* — [arXiv:2510.02384](https://arxiv.org/abs/2510.02384)
- **Axis 2** — Zhao, Gunn, Christ et al., *SoK: Watermarking for AI-Generated Content* — [arXiv:2411.18479](https://arxiv.org/abs/2411.18479)
- **Axis 3** — Hou et al., *Detecting AI-Generated Video: A Vision-Language Dual-View Survey* — ACL Findings 2026 — [arXiv:2607.10787](https://arxiv.org/abs/2607.10787)

### Axis 1 — Signal / Frequency
- *Frequency-Aware Robustness Analysis of Deepfake Detection Models* — JAI, 2026.3 — [techscience.com/jai/v8n1/66558](https://www.techscience.com/jai/v8n1/66558/html)
- *Hybrid Spatial–Frequency Attention with EfficientNet* — Scientific Reports, 2026.4 — [nature.com/articles/s41598-026-46086-9](https://www.nature.com/articles/s41598-026-46086-9)
- *Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking* — ACM TOMMCCAP, 2026.3 — [doi.org/10.1145/3797266](https://doi.org/10.1145/3797266)
- *Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale* — ICLR 2026 — [openreview.net/forum?id=XD431fRCg6](https://openreview.net/forum?id=XD431fRCg6)
- *FreqNet: Frequency-Aware Deepfake Detection* — AAAI 2024 — [arXiv:2403.07240](https://arxiv.org/abs/2403.07240)
- *Dual Data Alignment (DDA)* — NeurIPS 2025 (training-data compression/spectrum alignment)

### Axis 2 — Provenance
- Zhao et al., *Invisible Image Watermarks Are Provably Removable Using Generative AI* — NeurIPS 2024 — [arXiv:2306.01953](https://arxiv.org/abs/2306.01953)
- Müller & Debus, *The Watermark Shortcut: How Provenance Marking Sabotages Audio Deepfake Detection* — [arXiv:2606.23335](https://arxiv.org/abs/2606.23335)
- Nemecek et al., *Authenticated Contradictions from Desynchronized Provenance and Watermarking* — CVPR 2026 Workshops (APAI)
- Meta, *Video Seal: Open and Efficient Video Watermarking* — [arXiv:2412.09492](https://arxiv.org/abs/2412.09492)
- Google DeepMind, *SynthID Detector* & Content Detection API — [blog.google](https://blog.google/innovation-and-ai/products/google-synthid-ai-content-detector/) · [InfoQ 2026.5](https://www.infoq.com/news/2026/05/google-synthid-content-detection/)
- C2PA — Coalition for Content Provenance and Authenticity — [c2pa.org](https://c2pa.org/)

### Axis 3 — Semantic & Physics
- Shuai et al., *When Detectors Forget Forensics: Blocking Semantic Shortcuts* — [arXiv:2603.09242](https://arxiv.org/abs/2603.09242)
- *Detecting AI-Generated Images via Distributional Deviations (MPFT/TAM)* — [arXiv:2601.03586](https://arxiv.org/abs/2601.03586)
- *MDMF: Micro-Defects Expose Macro-Fakes* — [arXiv:2605.09296](https://arxiv.org/abs/2605.09296)
- *Physics-Driven Spatiotemporal Modeling (NSG-VD)* — NeurIPS 2025
- *DeCoF: Generated Video Detection via Frame Consistency* — [arXiv:2402.02085](https://arxiv.org/abs/2402.02085)

### Fusion & Benchmarks
- *SpecSem-Net: Integrating Spectral and Semantic Features* — [arXiv:2605.17311](https://arxiv.org/abs/2605.17311)
- *GenVidBench* — [arXiv:2501.11340](https://arxiv.org/abs/2501.11340) · *DeMamba/GenVideo* — Chen, Hong, Huang et al., *Sci. China Inf. Sci.* 69, 162103 (2026), [DOI:10.1007/s11432-024-4894-0](https://doi.org/10.1007/s11432-024-4894-0) (정식 동료심사 출판본; 이전 arXiv:2405.19707 대체) · *RobustSora* — [arXiv:2512.10248](https://arxiv.org/abs/2512.10248)

---

> 🎓 **Academic Research Project** — Undergraduate graduation project, Dept. of Computer Engineering.
> This system reports probabilistic risk scores for research and education purposes and must not be used as sole evidence for legal or financial decisions.
