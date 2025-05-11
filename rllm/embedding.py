# rllm/embedding.py

import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    ▷ 단어 임베딩 레이어
       - 토큰 ID를 d_model 차원의 실수 벡터로 변환합니다.
       - 변환된 벡터에 √d_model 스케일링을 적용하여,
         학습 초기 단계에서의 그래디언트 안정성을 높입니다.
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
          d_model    : 모델 내 임베딩 및 토큰 표현의 차원 수
          vocab_size : 어휘(vocabulary)의 크기 (총 토큰 종류 수)
        """
        super(Embeddings, self).__init__()
        # nn.Embedding: (vocab_size) → (d_model) 로 매핑하는 lookup 테이블
        self.lut = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=d_model)
        # 이후 스케일링 시에 사용할 차원 수 저장
        self.d_model = d_model

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
          x : (batch_size, seq_len) 형태의 토큰 ID 텐서

        Returns:
          (batch_size, seq_len, d_model) 형태의 임베딩 벡터
        """
        # 1) lut(x): 각 토큰 ID → d_model 차원의 실수 벡터
        # 2) * sqrt(d_model): 벡터 크기를 √d_model 만큼 스케일링
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    ▷ 사인·코사인 포지셔널 인코딩 (원본 논문 방식)
       - 토큰 순서 정보(sequence order)를 임베딩에 주입하기 위해 사용
       - max_length 까지의 위치별 인코딩을 미리 계산한 뒤,
         입력 시퀀스 길이에 맞게 슬라이스하여 더해 줍니다.
    """
    def __init__(self,
                 d_model: int = 512,
                 dropout: float = 0.1,
                 max_length: int = 5000):
        """
        Args:
          d_model    : 모델 차원 (Embeddings와 동일)
          dropout    : 드롭아웃 비율
          max_length : 미리 생성할 최대 시퀀스 길이
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        # 드롭아웃 레이어: overfitting 방지
        self.dropout = nn.Dropout(p=dropout)

        # 1) (1, max_length, d_model) 모양의 0 텐서 생성
        #    첫 차원이 배치 차원이 되도록 1로 설정
        pe = torch.zeros(1, max_length, d_model)

        # 2) 위치 인덱스 벡터: [0, 1, 2, ..., max_length-1] → shape (max_length, 1)
        position = torch.arange(0, max_length).unsqueeze(1).float()

        # 3) 주기(period) 벡터 계산:
        #    - 2i 차원: sin(position / (10000^(2i/d_model)))
        #    - 2i+1 차원: cos(position / (10000^(2i/d_model)))
        #    div_term[k] = exp(-ln(10000) * k / d_model) for k even
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # 4) pe 텐서에 사인·코사인 값 채우기
        pe[0, :, 0::2] = torch.sin(position * div_term)   # 짝수 인덱스
        pe[0, :, 1::2] = torch.cos(position * div_term)   # 홀수 인덱스

        # 5) buffer로 등록: state_dict에 저장되지만, 학습되지 않음
        #    to(device) 시 자동 이동됨
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x : (batch_size, seq_len, d_model) 형태의 임베딩 벡터

        Returns:
          (batch_size, seq_len, d_model) 형태로,
          입력 임베딩에 위치 인코딩을 더한 후 드롭아웃 적용
        """
        # 입력 시퀀스 길이
        seq_len = x.size(1)
        # 1) max_length 초과 시 예외 처리
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"[PositionalEncoding] 지원하는 max_length={self.pe.size(1)} 보다 "
                f"긴 시퀀스(seq_len={seq_len})입니다."
            )

        # 2) 입력 임베딩도 √d_model 스케일링
        #    (원 논문에서는 Embeddings 쪽에서만 스케일링하지만,
        #     전처리 과정 중 한 번 더 적용해도 무방)
        x = x * math.sqrt(self.d_model)

        # 3) 미리 계산된 pe에서 앞 seq_len 만큼 슬라이스하여 더하기
        #    pe: (1, max_length, d_model) → pe[:, :seq_len, :]
        pos_encoded = self.pe[:, :seq_len, :].to(x.device)
        x = x + pos_encoded

        # 4) 드롭아웃 적용 후 반환
        return self.dropout(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    텐서의 절반을 회전시킵니다.
    마지막 차원을 반으로 나누어 앞/뒤 부분을 바꿔줍니다.
    
    Args:
        x: 입력 텐서, 마지막 차원이 회전 대상
        
    Returns:
        회전된 텐서 (원본과 동일한 형태)
    """
    # 입력 텐서의 마지막 차원을 반으로 나눔
    x1 = x[..., : x.shape[-1] // 2]  # 앞쪽 절반
    x2 = x[..., x.shape[-1] // 2 :]  # 뒤쪽 절반
    
    # x2를 음수로 바꾸고 x1과 순서 바꿔서 다시 이어붙임
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionEmbedding(nn.Module):
    """
    ▷ Rotary Position Embedding (RoPE) 구현
       - LLaMA, GPT-NeoX 등 최신 모델에서 사용하는 위치 인코딩 방식
       - 토큰 위치에 따라 임베딩 벡터를 회전시켜 상대적 위치 정보를 반영
       - 장점: 토큰 간 상대적 위치 관계를 더 잘 학습하고, 더 긴 시퀀스로 확장 가능
    """
    def __init__(self, 
                 d_model: int, 
                 max_length: int = 5000,
                 base: int = 10000):
        """
        Args:
            d_model: 모델 차원 (반드시 짝수여야 함)
            max_length: 지원할 최대 시퀀스 길이
            base: 회전 주기 계산에 사용하는 기본값 (일반적으로 10000)
        """
        super(RotaryPositionEmbedding, self).__init__()
        
        # d_model이 짝수인지 확인
        assert d_model % 2 == 0, "d_model은 RoPE에서 반드시 짝수여야 합니다."
        self.d_model = d_model
        self.max_length = max_length
        
        # 회전에 사용할 프리컴퓨팅된 코사인과 사인 값 생성
        # 각 위치와 주파수 쌍에 대한 cos, sin 값 계산
        # shape: [max_length, d_model/2]
        self._precompute_freqs(d_model, max_length, base)
        
    def _precompute_freqs(self, dim: int, max_length: int, base: int = 10000):
        """
        위치 인코딩에 사용할 주파수 테이블을 미리 계산합니다.
        
        Args:
            dim: 모델 차원 (RoPE는 dim/2 크기의 sin/cos 쌍 사용)
            max_length: 최대 시퀀스 길이
            base: 스케일링 베이스 (일반적으로 10000)
        """
        # 차원 인덱스에 따른 주파수: (d_model/2,)
        # 각 쌍별 다른 주파수 사용
        freqs = torch.arange(0, dim, 2).float() / dim
        inv_freq = base ** -freqs  # (dim/2,)
        
        # 위치 인덱스: (max_length,)
        seq_idx = torch.arange(max_length).float()
        
        # 외적: (max_length, dim/2)
        sinusoid_inp = torch.outer(seq_idx, inv_freq)
        
        # sin, cos 계산: (max_length, dim/2)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        # 모델 생애 주기 동안 고정된 값 (학습되지 않음)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None, unsqueeze_dim: int = 1):
        """
        쿼리와 키 텐서에 회전 위치 인코딩을 적용합니다.
        
        Args:
            q: (batch_size, ..., seq_len, d_model) 형태의 쿼리 텐서
            k: (batch_size, ..., seq_len, d_model) 형태의 키 텐서
            position_ids: 위치 ID 텐서. None이면 0부터 seq_len-1까지의 인덱스 사용
            unsqueeze_dim: sin/cos를 unsqueeze할 차원 (어텐션 헤드 차원)
            
        Returns:
            회전 인코딩이 적용된 (q, k) 쌍
        """
        seq_len = q.size(-2)
        
        # position_ids가 없으면 0부터 seq_len-1까지의 인덱스 생성
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=q.device).unsqueeze(0)
        
        # 필요한 위치의 sin, cos 값만 가져오기
        sin = self.sin[position_ids]  # (batch_size, seq_len, dim/2)
        cos = self.cos[position_ids]  # (batch_size, seq_len, dim/2)
        
        # 차원에 따라 브로드캐스팅 가능하도록 unsqueeze
        sin = sin.unsqueeze(unsqueeze_dim)
        cos = cos.unsqueeze(unsqueeze_dim)
        
        # q, k에 회전 적용
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        # 원본 텐서와 동일한 dtype 유지
        return q_embed.type_as(q), k_embed.type_as(k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RoPE는 일반적으로 어텐션 계층에서 q, k에 직접 적용됩니다.
        이 메서드는 이전 PositionalEncoding과의 호환성을 위해 제공되지만,
        실제로는 어텐션 계층에서 apply_rotary_pos_emb 메서드를 직접 호출해야 합니다.
        
        Args:
            x: (batch_size, seq_len, d_model) 형태의 입력 텐서
            
        Returns:
            동일한 형태의 출력 텐서 (호환성을 위해 그대로 반환)
        """
        # 호환성을 위해 입력을 그대로 전달
        # 실제 RoPE는 MultiHeadAttention 내부에서 쿼리와 키에 적용됨
        return x
        
    def __repr__(self):
        return f"RotaryPositionEmbedding(d_model={self.d_model}, max_length={self.max_length})"
