# 한국어 LLM 모델 구현 및 학습 가이드

## 목차
1. [소개](#1-소개)
2. [이론적 배경](#2-이론적-배경)
3. [코드 구조 및 아키텍처](#3-코드-구조-및-아키텍처)
4. [데이터 처리 파이프라인](#4-데이터-처리-파이프라인)
5. [모델 구현](#5-모델-구현)
6. [학습 방법론](#6-학습-방법론)
7. [최적화 기법](#7-최적화-기법)
8. [실험 및 평가](#8-실험-및-평가)
9. [참고 문헌](#9-참고-문헌)

## 1. 소개

### 1.1 프로젝트 개요

이 문서는 한국어 텍스트 데이터를 활용한 언어 모델(LLM) 학습 프로젝트에 대한 포괄적인 가이드입니다. 본 프로젝트는 현대적인 트랜스포머 아키텍처를 기반으로 한국어 특성에 최적화된 언어 모델을 구현하고 학습하는 방법을 설명합니다.

### 1.2 주요 기술 스택

- **프레임워크**: PyTorch, PyTorch Lightning
- **최적화 기법**: Flash Attention, DeepSpeed
- **포지셔널 인코딩**: RoPE(Rotary Position Embedding)
- **정규화**: RMSNorm
- **데이터셋**: Huggingface의 한국어 교과서 데이터셋("maywell/korean_textbooks")
- **모니터링**: Weights & Biases (wandb)

### 1.3 한국어 데이터셋 특성

한국어는 교착어로서 영어 등의 굴절어와는 다른 언어적 특성을 갖습니다. 형태소의 결합이 복잡하고 어순이 상대적으로 자유로우며, 문맥에 따라 생략이 많습니다. 이러한 특성을 고려하여 토큰화 및 모델링 접근 방식을 적용했습니다.

본 프로젝트에서는 교과서 데이터셋을 활용하여 정제된 한국어에 대한 학습을 진행합니다. 교과서 텍스트는 문법적으로 정확하고 구조화된 내용을 포함하고 있어 기본 언어 이해에 적합합니다.

## 2. 이론적 배경

### 2.1 어텐션 메커니즘

#### 2.1.1 셀프 어텐션 (Self-Attention)

셀프 어텐션은 시퀀스 내 토큰들 간의 관계를 모델링하는 메커니즘으로, 입력 시퀀스의 각 요소가 다른 모든 요소와 어떻게 관련되는지를 계산합니다.

**수학적 정의:**

셀프 어텐션은 다음과 같이 정의됩니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서:
- $Q$ (Query): 입력 벡터에 가중치 행렬 $W^Q$를 곱한 결과
- $K$ (Key): 입력 벡터에 가중치 행렬 $W^K$를 곱한 결과
- $V$ (Value): 입력 벡터에 가중치 행렬 $W^V$를 곱한 결과
- $d_k$: 키 벡터의 차원, 스케일링 요소로 사용됨

#### 2.1.2 멀티헤드 어텐션 (Multi-Head Attention)

멀티헤드 어텐션은 입력 시퀀스를 여러 "헤드"에서 병렬적으로 처리하여 서로 다른 관점에서의 정보를 파악할 수 있게 합니다.

**수학적 정의:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

여기서 각 헤드는:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

멀티헤드 어텐션은 각 헤드가 입력 시퀀스의 다른 측면에 집중할 수 있도록 하여 더 풍부한 표현을 학습합니다.

#### 2.1.3 Flash Attention

Flash Attention은 2022년에 소개된 효율적인 어텐션 계산 알고리즘입니다. 기존 어텐션 구현의 메모리 병목 현상을 해결하기 위해 설계되었습니다.

**주요 특징:**
- IO 인식 알고리즘으로, GPU 메모리 계층 구조를 고려
- 어텐션 행렬을 명시적으로 저장하지 않음
- 블록 단위 처리로 메모리 접근 최적화
- 긴 시퀀스 처리에서 특히 효과적

Flash Attention은 메모리 소비를 $O(N^2)$에서 $O(N)$으로 줄여, 긴 시퀀스를 효율적으로 처리할 수 있게 합니다.

### 2.2 포지셔널 인코딩

#### 2.2.1 기존 사인-코사인 포지셔널 인코딩

트랜스포머는 순환 구조가 없기 때문에, 토큰의 위치 정보를 주입하기 위해 포지셔널 인코딩이 필요합니다. 원래 트랜스포머에서는 사인-코사인 함수 기반 인코딩을 사용했습니다:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

이 방식은 절대적 위치 정보를 모델에 제공합니다.

#### 2.2.2 RoPE(Rotary Position Embedding)

RoPE는 상대적 위치 정보를 인코딩하는 보다 효과적인 방법입니다. 회전 행렬을 사용하여 위치 정보를 임베딩합니다.

**수학적 정의:**

RoPE는 복소수 형태로 다음과 같이 정의됩니다:

$$\mathbf{q}'_m = \mathbf{q}_m e^{im\theta_j} = \mathbf{q}_m (\cos m\theta_j + i\sin m\theta_j)$$

실제 구현에서는 2D 회전 행렬을 사용하여 각 차원 쌍에 적용됩니다:

$$R_{\theta_j} = \begin{pmatrix} \cos \theta_j & -\sin \theta_j \\ \sin \theta_j & \cos \theta_j \end{pmatrix}$$

**RoPE의 장점:**
- 상대적 위치 인식 능력 향상
- 시퀀스 길이 확장 가능성 (extrapolation capacity)
- 회전 연산이 어텐션 계산과 잘 호환됨
- 컨텍스트 길이를 효과적으로 확장할 수 있음

### 2.3 정규화 기법

#### 2.3.1 LayerNorm

LayerNorm은 시퀀스의 각 위치별로 정규화를 수행하는 기법입니다:

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

여기서:
- $\mu$: 평균
- $\sigma^2$: 분산
- $\gamma, \beta$: 학습 가능한 스케일 및 시프트 파라미터
- $\epsilon$: 수치 안정성을 위한 작은 상수

#### 2.3.2 RMSNorm

RMSNorm은 LayerNorm의 계산을 단순화한 버전으로, 평균 계산을 생략하고 RMS(Root Mean Square)만을 사용합니다:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma$$

**RMSNorm의 장점:**
- 계산 효율성 향상 (평균 계산 단계 생략)
- 학습 안정성 유지
- 추론 속도 향상
- 일부 실험에서 더 나은 성능 보고

### 2.4 트랜스포머 아키텍처

#### 2.4.1 인코더-디코더 구조

원래의 트랜스포머는 인코더-디코더 구조로 설계되었습니다:

- **인코더**: 자기 주의 메커니즘과 피드포워드 네트워크로 구성된 여러 레이어를 통해 입력 시퀀스를 처리합니다.
- **디코더**: 인코더의 출력과 자신의 이전 출력을 바탕으로 다음 토큰을 예측합니다. 마스크드 자기 주의와 인코더-디코더 주의 메커니즘을 포함합니다.

이 구조는 주로 번역과 같은 시퀀스-투-시퀀스 작업에 사용됩니다.

#### 2.4.2 디코더 전용 LLM 설계

최근의 대부분의 LLM(GPT 계열, LLaMA, Gemma 등)은 디코더 전용 아키텍처를 채택하고 있습니다:

- 인코더 부분 제외
- 자기회귀적(autoregressive) 생성에 최적화
- 오직 마스크드 자기 주의만 사용

이 설계는 텍스트 생성 작업에 특화되어 있으며, 본 프로젝트에서도 디코더 전용 모델을 구현했습니다.

## 3. 코드 구조 및 아키텍처

### 3.1 프로젝트 구조

본 프로젝트는 모듈화된 구조로 설계되어 있으며, 각 파일이 특정 기능을 담당합니다:

```
rllm/
├── __init__.py            # 패키지 초기화
├── dataloader.py          # 데이터 로딩 및 처리
├── embedding.py           # 임베딩 및 RoPE 구현
├── mha.py                 # 멀티헤드 어텐션 구현
├── encoder.py             # 인코더 구현
├── decoder.py             # 디코더 구현
├── model.py               # 모델 클래스 정의
├── transformer.py         # 트랜스포머 아키텍처 구현
├── training.py            # 훈련 관련 유틸리티
├── trainer.py             # 훈련 로직 구현
├── train_korean.py        # 한국어 데이터 학습 스크립트
└── requirements.txt       # 의존성 패키지 목록
```

### 3.2 주요 모듈 관계

![모듈 관계도](https://mermaid.ink/img/pako:eNqFksFOwzAMhl_FuIBUCdZt06R2Jw4DwQHEYRdOURp3JVodV0m6MdS9O2kZGwjBLbH93_78x-0JCqOQFNCwbXAu3j7rPXO1Y9X4-lq4fYOtNXPB8lRU0rqqbJnE0gnnIB5_8Z9jVm-VtIJy5P-JX0nvkHmEOA8mJGc1HiKMk3iOABGX0lF-goTHCIPu_CuMhXKOW-OQvKY1QfzIWz9AH0cNM1FiHICrLl1Z0kuxbHRr-RBEuBP76EEYqzSBHbZK14RfJEXvpOxWJ-9p3jur2R7EVCZ5kPeBuQPxnOIBM24wfx1m3Kt-eLpN1KYh-3JW2w6E-2o9tPz8b_bO_e6YVl7nxMbsGxPg9vjt2d-ZW-9s76GXRFdhzQoY3XtIEAI2nTVFD1k0G0bR7ZhepRjRJIrTcDSc3twm0TROsnGWTu-i0WSwglb0ZSsR0gtKF2lWlJKc0fJUcNK26RnbNlkZ5w_PnZ4kh2sBM9LCRV5QYWihXK01u5JldN0LIVkpYbqbfQHkFtGE?type=png)

### 3.3 핵심 클래스 구조

#### 3.3.1 데이터 관련 클래스
- `CustomDataset`: 일반적인 데이터셋 처리
- `StreamingDataset`: 스트리밍 방식의 대용량 데이터셋 처리

#### 3.3.2 모델 관련 클래스
- `TransformerLightningModule`: PyTorch Lightning 기반 모델 클래스
- `Transformer`: 트랜스포머 구현
- `MultiHeadAttention`: 멀티헤드 어텐션 구현
- `PositionalEncoding`: 포지셔널 인코딩 구현
- `RotaryEmbedding`: RoPE 구현

#### 3.3.3 훈련 관련 클래스
- `NoamOpt`: Noam 스케줄러 구현
- `LabelSmoothing`: 레이블 스무딩 구현

## 4. 데이터 처리 파이프라인

### 4.1 데이터셋 로드

한국어 교과서 데이터셋은 Huggingface의 Datasets 라이브러리를 통해 스트리밍 방식으로 로드됩니다:

```python
from datasets import load_dataset

ds_train = load_dataset(
    "maywell/korean_textbooks",
    data_files="mmlu_high_school_statistics/train-00000-of-00001.parquet",
    split="train",
    streaming=True,
)
```

스트리밍 모드는 대용량 데이터셋을 처리할 때 메모리 효율성을 높입니다. 데이터를 한 번에 모두 메모리에 로드하지 않고, 필요에 따라 점진적으로 로드합니다.

### 4.2 토큰화 및 전처리

데이터는 Huggingface의 토크나이저를 통해 처리됩니다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

특수 토큰(PAD, EOS, BOS 등)이 적절히 설정되었는지 확인하고, 필요한 경우 추가합니다.

### 4.3 청크 생성 전략

긴 텍스트는 슬라이딩 윈도우 방식으로 고정 길이 청크로 분할됩니다:

```python
def _process_document(self, text: str):
    # 토큰화
    tokens = self.tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    
    # 슬라이딩 윈도우로 청크 생성
    for start in range(0, max(1, len(tokens) - self.max_length + 1), self.stride):
        # 청크 추출 및 처리
        # ...
```

주요 로직:
1. 전체 텍스트를 토큰화
2. `stride` 간격으로 시작 위치를 이동하며 `max_length` 크기의 청크 생성
3. 각 청크에 대해 입력 ID, 레이블, 어텐션 마스크 생성

### 4.4 인과적 언어 모델링 준비

각 청크는 인과적 언어 모델링(CLM)을 위해 다음과 같이 처리됩니다:

```python
# 인과적 언어 모델링을 위한 입력과 레이블 생성
# 입력: [토큰0, 토큰1, ..., 토큰N-1]
# 레이블: [토큰1, 토큰2, ..., 토큰N]
input_ids = current_slice[:-1]
labels = current_slice[1:]
```

이는 모델이 각 위치에서 다음 토큰을 예측하도록 훈련합니다.

### 4.5 배치 처리 및 버퍼링

스트리밍 데이터셋의 효율적인 처리를 위해 버퍼링 메커니즘을 구현했습니다:

```python
def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
    buffer = []
    
    # 데이터셋에서 샘플을 가져와 버퍼에 추가
    for sample in self.dataset:
        # ...
        chunks = self._process_document(text)
        buffer.extend(chunks)
        
        # 버퍼가 차면 셔플하고 반환
        if len(buffer) >= self.buffer_size:
            np.random.shuffle(buffer)
            for chunk in buffer:
                yield chunk
            buffer = []
```

버퍼링의 주요 이점:
1. 메모리 효율성: 한 번에 모든 데이터를 로드하지 않음
2. 셔플링 가능: 버퍼 내에서 샘플을 섞어 학습 효과 개선
3. 청크 크기 조정: 다양한 길이의 텍스트를 일정한 크기로 처리

### 4.6 데이터로더 구성

데이터로더는 배치 크기, 워커 수 등의 설정을 통해 효율적인 데이터 공급을 담당합니다:

```python
loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=shuffle and not is_streaming,  # 스트리밍일 때는 False
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    collate_fn=ds.collate_fn if hasattr(ds, 'collate_fn') else None,
    persistent_workers=num_workers > 0,
)
```

주요 설정:
- `pin_memory`: GPU 학습 시 메모리 전송 최적화
- `persistent_workers`: 워커 재사용으로 효율성 향상
- `collate_fn`: 커스텀 배치 생성 로직 적용

## 5. 모델 구현

### 5.1 TransformerLightningModule

`TransformerLightningModule` 클래스는 PyTorch Lightning 프레임워크를 활용해 트랜스포머 모델을 구현합니다. 이 클래스는 모델 정의, 훈련 및 평가 로직을 캡슐화합니다.

```python
class TransformerLightningModule(pl.LightningModule):
    def __init__(self,
                 src_vocab: int,
                 tgt_vocab: int,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_length: int = 5000,
                 pad_idx: int = 0,
                 lr_factor: float = 2.0,
                 warmup_steps: int = 4000,
                 label_smoothing: float = 0.1,
                 use_rope: bool = True,
                 use_flash_attn: bool = True,
                 tokenizer = None):
        super().__init__()
        # 모델 초기화...
```

주요 기능:
- 모델 아키텍처 설정 (차원, 레이어 수, 헤드 수 등)
- 최적화 설정 (학습률, 웜업 스텝 등)
- 훈련/검증/테스트 단계 정의
- 로깅 및 체크포인트 관리

### 5.2 멀티헤드 어텐션 구현

`mha.py` 파일에는 멀티헤드 어텐션 메커니즘의 구현이 포함되어 있습니다:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_flash_attn=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.use_flash_attn = use_flash_attn
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 선형 투영 및 헤드 분할
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Flash Attention 사용 여부에 따라 다른 어텐션 계산
        if self.use_flash_attn and flash_attn_available:
            # Flash Attention 구현
            attn_output = flash_attention(q, k, v, mask)
        else:
            # 기존 스케일드 닷-프로덕트 어텐션
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # 헤드 결합 및 최종 선형 투영
        output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)
```

주요 부분:
- 쿼리, 키, 밸류 벡터 생성을 위한 선형 레이어
- 헤드 분할 및 결합 로직
- Flash Attention 지원 (사용 가능한 경우)
- 어텐션 스코어 계산 및 소프트맥스 적용

### 5.3 RoPE 구현

`embedding.py` 파일에는 RoPE(Rotary Position Embedding)의 구현이 포함되어 있습니다:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 캐시 초기화
        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        # 캐시된 값이 있고 시퀀스 길이가 같으면 캐시 사용
        if self._rotary_pos_emb_cache is not None and seq_len == self._seq_len_cached:
            return self._rotary_pos_emb_cache
            
        # 포지션 임베딩 계산
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 복소수 형태로 변환 (cos, sin 값)
        cos_cached = emb.cos()[:, None, None, :]
        sin_cached = emb.sin()[:, None, None, :]
        
        # 캐시 업데이트
        self._rotary_pos_emb_cache = (cos_cached, sin_cached)
        self._seq_len_cached = seq_len
        
        return self._rotary_pos_emb_cache
        
# RoPE 적용 함수
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # q, k 형태: [batch_size, seq_len, num_heads, head_dim]
    
    # 짝수 및 홀수 차원을 분리
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # 회전 적용
    q_rotated = torch.cat([
        q_even * cos - q_odd * sin,
        q_odd * cos + q_even * sin
    ], dim=-1)
    
    k_rotated = torch.cat([
        k_even * cos - k_odd * sin,
        k_odd * cos + k_even * sin
    ], dim=-1)
    
    return q_rotated, k_rotated
```

RoPE의 핵심 부분:
1. 각 위치에 대한 회전 각도 계산
2. 코사인과 사인 값 생성 및 캐싱
3. 쿼리와 키 벡터에 회전 적용

### 5.4 RMSNorm 구현

RMSNorm은 LayerNorm의 변형으로, 평균 계산을 생략하고 계산을 단순화합니다:

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x):
        # 평균이 아닌 제곱평균제곱근만 사용
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.scale
```

RMSNorm의 이점:
- 계산 효율성: 평균 계산을 생략하여 연산 감소
- 단순한 구현: 코드가 더 간결해짐
- 성능 향상: 일부 경우에 LayerNorm보다 나은 성능 제공

### 5.5 인코더-디코더 구조

본 프로젝트에서는 언어 모델링을 위해 주로 디코더 부분을 활용하지만, 인코더-디코더 아키텍처도 구현되어 있습니다:

```python
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1, use_rope=False, use_flash_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout, use_rope, use_flash_attn)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model) if use_rope else nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1, use_rope=False, use_flash_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout, use_rope, use_flash_attn)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model) if use_rope else nn.LayerNorm(d_model)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

이 구조는 각각 여러 층의 인코더/디코더 레이어로 구성되며, 각 레이어는 어텐션 메커니즘과 피드포워드 네트워크를 포함합니다.

## 6. 학습 방법론

### 6.1 PyTorch Lightning 프레임워크

PyTorch Lightning은 PyTorch 기반의 고수준 인터페이스로, 연구 코드를 효율적으로 구성하고 확장성 있게 만듭니다:

```python
# 트레이너 설정
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=gpus,
    strategy=strategy,
    precision=precision,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, lr_monitor],
    accumulate_grad_batches=accumulate_grad_batches,
    gradient_clip_val=gradient_clip_val,
    log_every_n_steps=10,
)

# 학습 시작
trainer.fit(model, train_dataloaders=train_loader)
```

Lightning의 이점:
- 보일러플레이트 코드 감소
- 분산 학습 간소화
- 하드웨어 추상화 (CPU/GPU/TPU)
- 코드 구성 표준화

### 6.2 학습률 스케줄링 (Noam Optimizer)

트랜스포머 모델 학습에는 Noam 스케줄러가 효과적입니다:

```python
class NoamOpt:
    """Optimizer with Noam learning rate schedule"""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        """Implement learning rate schedule"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) * 
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
```

Noam 스케줄러의 특징:
- 웜업 기간 동안 학습률을 선형적으로 증가
- 이후 제곱근에 반비례하여 서서히 감소
- 모델 차원에 맞춰 자동 스케일링

### 6.3 레이블 스무딩 기법

레이블 스무딩은 모델의 과적합을 방지하고 일반화 성능을 향상시키는 기법입니다:

```python
class LabelSmoothing(nn.Module):
    """Label smoothing loss function"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
```

레이블 스무딩의 이점:
- 모델 일반화 능력 향상
- 과신뢰(overconfidence) 방지
- 정규화 효과로 과적합 감소

### 6.4 Gradient Accumulation 및 Clipping

효율적인 학습을 위해 그래디언트 누적 및 클리핑을 적용합니다:

```python
# 트레이너 설정에서 그래디언트 누적 및 클리핑 설정
trainer = pl.Trainer(
    # ...
    accumulate_grad_batches=accumulate_grad_batches,  # 그래디언트 누적
    gradient_clip_val=gradient_clip_val,              # 그래디언트 클리핑
    # ...
)
```

이 기법들의 이점:
- **그래디언트 누적**: 더 큰 가상 배치 크기로 학습 가능
- **그래디언트 클리핑**: 학습 안정성 향상 및 exploding gradient 방지

### 6.5 체크포인트 저장 전략

학습 중 주기적으로 모델을 저장하고, 최적의 모델을 추적합니다:

```python
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="{epoch:02d}-{step}-{train_loss:.4f}",
    save_top_k=3,              # 최고 성능 모델 3개 저장
    monitor="train_loss",      # 손실 기준으로 모델 선택
    mode="min",                # 손실 최소화
    every_n_train_steps=save_every_n_steps,  # 지정된 스텝마다 저장
    save_last=True,            # 마지막 모델도 저장
)
```

체크포인트 전략의 이점:
- 학습 중단 시에도 복구 가능
- 최적의 모델 추적 및 보존
- 실험 재현성 제고

## 7. 최적화 기법

### 7.1 Flash Attention 구현 및 활용

Flash Attention은 어텐션 계산을 위한 메모리 효율적인 알고리즘입니다:

```python
def flash_attention(q, k, v, mask=None, dropout_p=0.0):
    """
    Flash Attention 구현
    입력:
      q, k, v: 쿼리, 키, 밸류 텐서 [배치크기, 헤드수, 시퀀스길이, 헤드차원]
      mask: 어텐션 마스크
      dropout_p: 드롭아웃 확률
    """
    # flash_attn 라이브러리 사용
    from flash_attn import flash_attn_func
    
    # 형태 변환 (Flash Attention은 특정 입력 형태 요구)
    batch_size, num_heads, seq_len, head_dim = q.shape
    q, k, v = [x.transpose(1, 2).reshape(batch_size, seq_len, num_heads, head_dim) for x in (q, k, v)]
    
    # 마스크 처리
    if mask is not None:
        # Flash Attention의 마스크 형식에 맞게 변환
        # ...
    
    # Flash Attention 함수 호출
    output = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=mask is not None)
    
    # 출력 형태 원복
    output = output.reshape(batch_size, seq_len, num_heads * head_dim).transpose(1, 2).reshape(batch_size, num_heads, seq_len, head_dim)
    
    return output
```

Flash Attention의 주요 이점:
- **메모리 효율성**: O(N²) → O(N) 메모리 복잡도 감소
- **계산 최적화**: IO 인식 알고리즘으로 속도 향상
- **긴 시퀀스 지원**: 더 긴 컨텍스트 길이 처리 가능

### 7.2 DeepSpeed 통합

DeepSpeed는 대규모 모델 학습을 위한 최적화 라이브러리로, 다양한 병렬화 전략을 제공합니다:

```python
# DeepSpeed 전략 설정
if gpus > 1:
    strategy = DeepSpeedStrategy(
        stage=2,                       # ZeRO-2 최적화
        offload_optimizer=True,        # 옵티마이저 상태를 CPU로 오프로드
        offload_parameters=False,      # 파라미터는 GPU에 유지
        allgather_bucket_size=5e8,     # 버킷 크기 최적화
        reduce_bucket_size=5e8,
    )
else:
    strategy = "auto"
```

DeepSpeed의 주요 기능:
- **ZeRO (Zero Redundancy Optimizer)**: 모델 병렬화의 효율성 향상
- **오프로딩**: CPU 메모리를 활용한 GPU 메모리 부담 감소
- **그래디언트 누적**: 효과적인 대규모 배치 학습

### 7.3 메모리 최적화 기법

대규모 모델 학습 시 메모리 사용을 최적화하기 위한 여러 기법이 적용되었습니다:

```python
# 메모리 효율적인 어텐션 계산
def memory_efficient_attention(q, k, v, mask=None):
    # 청크 단위 처리로 메모리 사용량 최적화
    chunk_size = 1024
    output_chunks = []
    
    for i in range(0, q.shape[2], chunk_size):
        q_chunk = q[:, :, i:i+chunk_size]
        # 어텐션 계산
        chunk_output = scaled_dot_product_attention(q_chunk, k, v, mask)
        output_chunks.append(chunk_output)
    
    return torch.cat(output_chunks, dim=2)
```

추가 메모리 최적화 기법:
- **그래디언트 체크포인팅**: 순방향 활성화를 저장하는 대신 역방향 계산 시 재계산
- **버퍼 재사용**: 중간 결과물을 위한 버퍼 재활용
- **정밀도 최적화**: 적절한 데이터 타입 사용 (FP16/BF16)

### 7.4 혼합 정밀도 학습 (Mixed Precision)

혼합 정밀도 학습은 모델 학습 속도를 높이고 메모리 사용량을 줄입니다:

```python
# PyTorch Lightning에서 혼합 정밀도 설정
trainer = pl.Trainer(
    # ...
    precision="bf16-mixed",  # BF16 혼합 정밀도 사용
    # ...
)
```

혼합 정밀도의 이점:
- **메모리 효율성**: FP16/BF16은 FP32 대비 메모리 절반 사용
- **계산 속도 향상**: 현대 GPU는 반정밀도 연산 가속 지원
- **학습 안정성**: 마스터 가중치는 FP32로 유지하여 안정성 확보

## 8. 실험 및 평가

### 8.1 Weights & Biases 모니터링

Weights & Biases(wandb)를 사용하여 학습 과정을 실시간으로 모니터링하고 기록합니다:

```python
# wandb 로거 설정
wandb_logger = WandbLogger(
    project=project_name,
    name=run_name,
    log_model="all",  # 모든 체크포인트 저장
)

# 하이퍼파라미터 로깅
wandb_logger.experiment.config.update({
    "d_model": d_model,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "max_length": max_length,
    "batch_size": batch_size,
    "lr_factor": lr_factor,
    "warmup_steps": warmup_steps,
    "use_rope": use_rope,
    "use_flash_attn": use_flash_attn,
    "tokenizer": model_name_or_path,
})
```

wandb 모니터링의 이점:
- **실시간 학습 추적**: 손실, 학습률, 메모리 사용량 등 실시간 모니터링
- **실험 비교**: 여러 실험 설정을 비교 분석
- **하이퍼파라미터 추적**: 설정 기록 및 최적 조합 탐색

### 8.2 주요 하이퍼파라미터 조정 전략

효과적인 학습을 위해 다음과 같은 하이퍼파라미터 조정 전략을 사용합니다:

| 하이퍼파라미터 | 권장 범위 | 조정 전략 |
|--------------|---------|---------|
| 학습률 | 1e-4 ~ 5e-4 | Noam 스케줄러 사용, 모델 크기에 따른 스케일링 |
| 배치 크기 | 8 ~ 64 | 가용 GPU 메모리에 맞춰 최대화 |
| 웜업 스텝 | 1000 ~ 5000 | 데이터셋 크기의 5~10% 정도 |
| 드롭아웃 | 0.1 ~ 0.2 | 모델 크기가 클수록 낮은 드롭아웃 |
| 레이블 스무딩 | 0.1 | 대부분의 경우 0.1이 적합 |

```python
# train_korean.py 예시
parser.add_argument("--d_model", type=int, default=768, help="모델 차원")
parser.add_argument("--d_ff", type=int, default=3072, help="피드포워드 레이어 차원")
parser.add_argument("--num_heads", type=int, default=12, help="어텐션 헤드 수")
parser.add_argument("--num_layers", type=int, default=12, help="레이어 수")
parser.add_argument("--dropout", type=float, default=0.1, help="드롭아웃 비율")
parser.add_argument("--max_length", type=int, default=1024, help="최대 시퀀스 길이")
parser.add_argument("--lr_factor", type=float, default=1.0, help="학습률 스케일링 계수")
```

### 8.3 성능 평가 지표

모델 성능을 평가하기 위한 주요 지표:

- **학습 손실**: 모델이 얼마나 잘 학습하고 있는지 보여주는 기본 지표
- **검증 손실**: 과적합 모니터링 및 일반화 성능 측정
- **PPL (퍼플렉서티)**: 언어 모델 성능의 표준 지표, 낮을수록 좋음
- **토큰당 처리 시간**: 학습 및 추론 효율성 평가
- **메모리 사용량**: 최적화 효과 평가

```python
def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    # 모델 forward 패스
    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    # 퍼플렉서티 계산
    ppl = torch.exp(loss)
    
    # 로깅
    self.log("train_loss", loss, prog_bar=True)
    self.log("train_ppl", ppl, prog_bar=True)
    
    return loss
```

### 8.4 학습 결과 시각화

wandb를 통해 다양한 시각화를 제공합니다:

```python
# 각 단계마다 주요 지표 로깅
def on_train_batch_end(self, outputs, batch, batch_idx):
    # 추가 메트릭 계산
    lr = self.trainer.optimizers[0].param_groups[0]["lr"]
    self.log("learning_rate", lr, prog_bar=True)
    
    # 주기적으로 샘플 생성 결과 로깅 (50 배치마다)
    if batch_idx % 50 == 0:
        sample_text = self.generate_sample()
        self.logger.experiment.log({
            "sample_generation": wandb.Html(f"<pre>{sample_text}</pre>")
        })
```

시각화 종류:
- **손실 곡선**: 학습 및 검증 손실 추이
- **학습률 곡선**: Noam 스케줄러의 학습률 변화
- **어텐션 맵**: 특정 샘플에 대한 어텐션 패턴
- **샘플 생성**: 학습 중 주기적인 텍스트 샘플 생성

## 9. 참고 문헌

### 9.1 트랜스포머 아키텍처

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

### 9.2 RoPE (Rotary Position Embedding)

2. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv preprint arXiv:2104.09864*.

### 9.3 Flash Attention

3. Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems*, 35.

### 9.4 RMSNorm

4. Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *Advances in Neural Information Processing Systems*, 32.

### 9.5 효율적인 학습 기법

5. Narang, S., Chung, H. W., Tay, Y., Fedus, W., Fevry, T., Matena, M., ... & Raffel, C. (2021). Do transformer modifications transfer across implementations and applications? *arXiv preprint arXiv:2102.11972*.

### 9.6 대규모 언어 모델

6. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in neural information processing systems*, 33, 1877-1901.

7. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*. 