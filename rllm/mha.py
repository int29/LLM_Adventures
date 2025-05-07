# rllm/mha.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor = None,
              dropout: nn.Dropout = None):
    """
    스케일된 닷-프로덕트 어텐션 함수 구현

    Args:
      query: (batch_size, num_heads, seq_len_q, d_k) 형태의 쿼리 텐서
      key:   (batch_size, num_heads, seq_len_k, d_k) 형태의 키 텐서
      value: (batch_size, num_heads, seq_len_v, d_v) 형태의 값 텐서
      mask:  어텐션 마스크 (padding 또는 subsequent mask)
             - (batch_size, 1, 1, seq_len_k) 또는
             - (batch_size, 1, seq_len_q, seq_len_k)
      dropout: 어텐션 확률에 적용할 Dropout 모듈

    Returns:
      output: (batch_size, num_heads, seq_len_q, d_v) 형태의 결과
      attn:   (batch_size, num_heads, seq_len_q, seq_len_k) 형태의 어텐션 확률
    """
    # d_k: 각 헤드마다의 쿼리/키 차원
    d_k = query.size(-1)
    # 1) 쿼리와 키의 내적 계산 및 d_k의 제곱근으로 스케일링
    #    shape: (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 2) 마스크 처리
        # mask가 3차원일 경우 (batch, seq_len_q, seq_len_k) 형태 → 4차원으로 확장
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_k)
        # 마스크를 모든 헤드에 동일하게 적용하기 위해 복제
        if mask.size(1) == 1 and scores.size(1) > 1:
            mask = mask.expand(-1, scores.size(1), -1, -1)
        # mask==0인 위치는 매우 작은 값으로 채워 softmax 이후 0이 되도록 함
        scores = scores.masked_fill(mask == 0, float('-1e9'))

    # 3) softmax로 어텐션 확률 계산
    attn = F.softmax(scores, dim=-1)
    # 4) Dropout 적용 (옵션)
    if dropout is not None:
        attn = dropout(attn)
    # 5) 어텐션 확률과 value 행렬 곱셈으로 최종 값 계산
    #    shape: (batch, heads, seq_len_q, d_v)
    output = torch.matmul(attn, value)
    return output, attn


class MultiHeadAttention(nn.Module):
    """
    멀티-헤드 어텐션 모듈 구현

    전체 d_model 차원을 여러 헤드로 분할해 병렬 어텐션을 수행한 뒤
    결과를 합쳐 다시 d_model 차원으로 프로젝션합니다.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1):
        """
        Args:
          d_model:   모델 차원 (전체)
          num_heads: 어텐션 헤드 개수
          dropout:   어텐션 확률에 적용할 드롭아웃 비율
        """
        super(MultiHeadAttention, self).__init__()
        # d_model이 num_heads로 나누어떨어지는지 확인
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 합니다."
        self.d_model = d_model
        self.num_heads = num_heads
        # 각 헤드 당 쿼리/키/값 차원
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 입력(전체 d_model) → 각 헤드 d_model 차원으로 선형 프로젝션
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # 여러 헤드를 concat 한 뒤 다시 d_model 차원으로 프로젝션
        self.W_o = nn.Linear(d_model, d_model)

        # 어텐션 확률에 적용할 드롭아웃 레이어
        self.dropout = nn.Dropout(p=dropout)
        # attention 가중치를 저장할 변수 (디버깅/시각화용)
        self.attn = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          query: (batch_size, seq_len_q, d_model)
          key:   (batch_size, seq_len_k, d_model)
          value: (batch_size, seq_len_v, d_model)
          mask:  어텐션 마스크

        Returns:
          (batch_size, seq_len_q, d_model) 형태의 멀티-헤드 어텐션 결과
        """
        batch_size = query.size(0)

        # 1) 입력을 Q, K, V로 선형 변환
        #    결과 shape: (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2) num_heads로 분할
        #    - view: (batch, seq_len, num_heads, d_k)
        #    - transpose: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # 3) scaled dot-product attention 호출
        #    output: (batch, num_heads, seq_len_q, d_v)
        #    self.attn: (batch, num_heads, seq_len_q, seq_len_k)
        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 4) 여러 헤드를 다시 합치기
        #    - transpose: (batch, seq_len_q, num_heads, d_v)
        #    - contiguous.view: (batch, seq_len_q, num_heads*d_v == d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5) 최종 선형 프로젝션
        output = self.W_o(x)
        return output
