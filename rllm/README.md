# LLM 프리트레이닝 프로젝트

대규모 언어 모델(LLM)을 처음부터 사전 학습하기 위한 파이토치 기반 프레임워크입니다. Transformer 아키텍처를 기반으로 하며, 최신 기술을 적용하여 효율적인 학습을 지원합니다.

## 주요 기능

- **Flash Attention**: 메모리 효율적인 어텐션 메커니즘으로 더 긴 시퀀스와 더 큰 배치 크기를 처리할 수 있습니다.
- **RoPE(Rotary Position Embedding)**: 위치 정보를 효과적으로 인코딩하는 최신 방식으로, LLaMA와 같은 최신 LLM에서 사용되는 기법입니다.
- **Huggingface 통합**: Gemma-3, Llama 등 최신 토크나이저 및 HF 라이브러리와의 완벽한 통합을 지원합니다.
- **인과적 언어 모델링(CLM)**: 현대적인 LLM 학습 방식을 적용한 효율적인 언어 모델링을 지원합니다.
- **PyTorch Lightning**: 코드 구조화와 분산 학습을 간소화합니다.
- **Fabric**: 단일 GPU에서 멀티 GPU/TPU까지 쉽게 확장 가능한 유연한 학습 인프라를 제공합니다.
- **효율적인 데이터 로딩**: 슬라이딩 윈도우 방식의 CustomDataset으로 대용량 텍스트 데이터를 효율적으로 처리합니다.
- **Noam 학습률 스케줄링**: Transformer 학습에 최적화된 학습률 스케줄링을 지원합니다.

## 설치 방법

```bash
git clone https://github.com/your-username/rllm.git
cd rllm
pip install -r requirements.txt
```

## 필요 패키지

```
torch>=2.1.0
pytorch-lightning>=2.0.0
flash-attention>=1.0.0
transformers>=4.37.0
datasets>=2.0.0
numpy>=1.20.0
tqdm>=4.60.0
tensorboard>=2.10.0
pyyaml>=5.4.0
fsspec>=2022.5.0
typing-extensions>=4.4.0
tokenizers>=0.15.0
einops>=0.7.0
accelerate>=0.25.0
```

## 프로젝트 구조

```
rllm/
├── dataloader.py      # 데이터 로딩 및 전처리 (Huggingface 토크나이저 통합)
├── embedding.py       # 임베딩 및 포지셔널 인코딩(RoPE 포함)
├── mha.py             # 멀티헤드 어텐션 (Flash Attention, RoPE 포함)
├── encoder.py         # Transformer 인코더 (RMSNorm 포함)
├── decoder.py         # Transformer 디코더
├── transformer.py     # 전체 Transformer 모델
├── training.py        # 학습 관련 유틸리티
├── model.py           # PyTorch Lightning 모듈
├── trainer.py         # 학습 스크립트 (Lightning/Fabric)
└── README.md          # 문서
```

## 주요 기술 설명

### 현대적인 LLM 데이터 전처리

최신 LLM 아키텍처에 맞춘 데이터 전처리 파이프라인:

- **Huggingface 토크나이저**: `google/gemma-3-1b-pt` 등 최신 토크나이저 지원으로 효율적인 토큰화.
- **인과적 언어 모델링(CLM)**: 입력과 레이블을 적절히 시프트하여 다음 토큰 예측 학습.
- **어텐션 마스크**: 패딩 토큰을 효과적으로 처리하는 마스크 생성.
- **레이블 마스킹**: `-100` 값으로 손실 계산 시 패딩 토큰을 무시.
- **슬라이딩 윈도우**: 긴 텍스트를 효율적으로 분할하는 윈도우 방식 적용.

### RoPE(Rotary Position Embedding)

RoPE는 토큰 간의 상대적 위치 정보를 보존하는 방식으로 어텐션 메커니즘을 강화합니다:

- 각 토큰의 위치에 따라 쿼리(Q)와 키(K) 벡터를 회전시킵니다.
- 절대적 위치 인코딩과 달리, 상대적 위치 정보가 더 자연스럽게 보존됩니다.
- 컨텍스트 길이 확장(context length extension)에 더 효과적입니다.

### Flash Attention

메모리 효율적인 어텐션 계산 방식으로, 시퀀스 길이에 따른 메모리 요구사항을 줄입니다:

- IO-aware 계산을 통해 메모리 대역폭 사용을 최적화합니다.
- 더 긴 시퀀스와 더 큰 배치 크기로 학습이 가능합니다.
- 특히 긴 컨텍스트 처리에 효과적입니다.

### RMSNorm

LayerNorm의 개선된 버전으로, 계산 효율성과 성능이 향상됩니다:

- 평균 제거(centering) 단계가 없어 계산이 더 효율적입니다.
- 특히 대규모 LLM에서 성능 향상이 두드러집니다.
- 수치 안정성과 일반화 성능에 긍정적 영향을 줍니다.

## 사용 방법

### 1. PyTorch Lightning으로 학습하기

```bash
python -m rllm.trainer --mode lightning --batch_size 32 --max_epochs 10 --d_model 512 --num_heads 8 --precision "16-mixed" --use_rope --tokenizer_name google/gemma-3-1b-pt
```

### 2. Fabric으로 학습하기

```bash
python -m rllm.trainer --mode fabric --batch_size 32 --max_epochs 10 --devices 1 --use_rope --use_flash_attn --tokenizer_name google/gemma-3-1b-pt
```

### 3. 다중 GPU로 확장하기

```bash
python -m rllm.trainer --mode fabric --devices 4 --precision "bf16-mixed" --use_rope --use_flash_attn --tokenizer_name google/gemma-3-1b-pt
```

### 4. 다른 토크나이저 사용하기

```bash
python -m rllm.trainer --mode fabric --tokenizer_name meta-llama/Llama-3-8B
```

## 주요 매개변수

- `--mode`: 학습 모드 선택 (`lightning` 또는 `fabric`)
- `--batch_size`: 배치 크기
- `--max_epochs`: 최대 에포크 수
- `--d_model`: 모델 내부 차원
- `--num_heads`: 어텐션 헤드 수
- `--num_layers`: 인코더/디코더 레이어 수
- `--max_length`: 최대 시퀀스 길이
- `--precision`: 연산 정밀도 (`32-true`, `16-mixed`, `bf16-mixed` 등)
- `--accelerator`: 가속기 타입 (`cpu`, `gpu`, `tpu`, `auto`)
- `--devices`: 사용할 디바이스 수
- `--ckpt_path`: 체크포인트 경로 (있을 경우)
- `--use_rope`: RoPE(Rotary Position Embedding) 사용 (기본값: True)
- `--no_rope`: RoPE 사용하지 않음
- `--use_flash_attn`: Flash Attention 사용 (기본값: True)
- `--no_flash_attn`: Flash Attention 사용하지 않음
- `--tokenizer_name`: Huggingface 토크나이저 이름 (기본값: "google/gemma-3-1b-pt")

## 예제 코드

### 모델 인스턴스 생성

```python
from rllm.model import TransformerLightningModule
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

model = TransformerLightningModule(
    src_vocab=len(tokenizer.vocab),
    tgt_vocab=len(tokenizer.vocab),
    d_model=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    use_rope=True,
    use_flash_attn=True,
    tokenizer=tokenizer
)
```

### 데이터 로더 생성

```python
from rllm.dataloader import create_dataloader

train_loader, tokenizer = create_dataloader(
    dataset=your_dataset,
    batch_size=32,
    max_length=1024,
    stride=512,
    tokenizer_name="google/gemma-3-1b-pt"
)
```

### 커스텀 데이터셋으로 학습

```python
from rllm.trainer import train_with_fabric

model, tokenizer = train_with_fabric(
    train_dataset=your_train_dataset,
    val_dataset=your_val_dataset,
    batch_size=32,
    max_epochs=10,
    d_model=512,
    num_heads=8,
    precision="bf16-mixed",
    use_rope=True,
    use_flash_attn=True,
    tokenizer_name="google/gemma-3-1b-pt"
)
```

### 텍스트 생성 추론

```python
# 모델 로드 (학습 후)
input_text = "인공지능이란"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

# 텍스트 생성
generated_ids = model.generate(
    input_ids=input_ids,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# 디코딩
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

## 성능 팁

- **bf16-mixed** 정밀도를 사용하면 학습 속도를 크게 향상시킬 수 있습니다.
- 가능하면 **Flash Attention**을 활성화하여 메모리 효율성과 속도를 개선하세요.
- 긴 컨텍스트 처리에는 **RoPE**가 효과적으로 도움을 줍니다.
- Huggingface의 **Gemma** 또는 **Llama** 토크나이저를 사용하면 효율적인 토큰화가 가능합니다.
- 배치 크기는 사용 가능한 GPU 메모리에 맞게 조정하세요.

## 라이센스

MIT 라이센스

## 참고 문헌

1. "Attention is All You Need" - Vaswani et al.
2. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" - Dao et al.
3. "RoFormer: Enhanced Transformer with Rotary Position Embedding" - Su et al.
4. "Root Mean Square Layer Normalization" - Zhang et al.
5. "LLaMA: Open and Efficient Foundation Language Models" - Touvron et al.
6. "Gemma: Open Models Based on Gemini Research and Technology" - Google 