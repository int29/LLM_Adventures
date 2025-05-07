# rllm/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from rllm.mha import MultiHeadAttention
from rllm.embedding import Embeddings, PositionalEncoding

class LayerNorm(nn.Module):
    """
    ▷ Transformer 논문에서 제안된 Layer Normalization
    ▷ 입력 텐서의 마지막 차원(feature 차원)에 대해 평균과 분산으로 정규화
    """
    def __init__(self, features: int, eps: float = 1e-6):
        """
        Args:
          features: 정규화할 feature 차원 크기 (d_model)
          eps:      수치 안정성을 위한 작은 상수
        """
        super(LayerNorm, self).__init__()
        # 스케일과 시프트를 위한 학습 가능한 파라미터
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, seq_len, features) 형태의 입력
        Returns:
          동일한 shape의 정규화된 출력
        """
        # 마지막 차원(feature) 기준으로 평균과 표준편차 계산
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        # 정규화 후 학습 가능한 a_2, b_2 적용
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """
    ▷ 각 토큰 위치마다 독립적으로 적용되는 2-layer Feed-Forward 네트워크
    ▷ 입력 차원 → 내부 d_ff 차원 → 출력 차원 으로 변환
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
          d_model: 모델 차원 (입력/출력 크기)
          d_ff:    내부 은닉층 크기
          dropout: 드롭아웃 비율
        """
        super(PositionwiseFeedForward, self).__init__()
        # 첫 번째 선형 변환: d_model → d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 두 번째 선형 변환: d_ff → d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # 활성화는 ReLU, 중간에 드롭아웃 적용
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, seq_len, d_model)
        Returns:
          (batch_size, seq_len, d_model)
        """
        # 1) 첫 선형 → ReLU → 드롭아웃
        # 2) 두 선형
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    ▷ 각 서브레이어(어텐션 또는 FFN) 앞뒤로
      1) LayerNorm → 2) 서브레이어 적용 → 3) Dropout → 4) Residual 연결
    """
    def __init__(self, size: int, dropout: float):
        """
        Args:
          size:    d_model 크기
          dropout: 드롭아웃 비율
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Args:
          x:        (batch_size, seq_len, d_model)
          sublayer: LayerNorm 이후 적용할 함수(예: self-attn or feed-forward)
        Returns:
          Residual + Dropout + SublayerNorm 결과
        """
        # 1) x에 LayerNorm 적용
        # 2) sublayer(x_norm) → 드롭아웃 → x + result (residual)
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    ▷ Transformer 인코더의 한 레이어 구성
    ▷ 두 개의 서브레이어:
       1) Self-Attention
       2) Position-wise Feed-Forward
    """
    def __init__(self,
                 size: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float):
        """
        Args:
          size:         d_model 크기
          self_attn:    멀티-헤드 셀프 어텐션 모듈
          feed_forward: 위치별 Feed-Forward 모듈
          dropout:      드롭아웃 비율
        """
        super(EncoderLayer, self).__init__()
        # 두 개의 sublayer 연결: [self-attn, feed-forward]
        self.sublayer = nn.ModuleList([
            SublayerConnection(size, dropout) for _ in range(2)
        ])
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          x:    (batch_size, seq_len, d_model)
          mask: 패딩 마스크 (batch_size, 1, 1, seq_len)
        Returns:
          (batch_size, seq_len, d_model)
        """
        # 1) 첫 번째 sublayer: self-attention
        x = self.sublayer[0](x, lambda x_norm: self.self_attn(x_norm, x_norm, x_norm, mask))
        # 2) 두 번째 sublayer: feed-forward
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    ▷ N개의 EncoderLayer를 쌓아서 완전한 인코더 블록을 구성
    ▷ 마지막에 추가 LayerNorm 적용
    """
    def __init__(self, layer: EncoderLayer, N: int):
        """
        Args:
          layer: 한 개의 EncoderLayer 인스턴스 (나중에 복제됨)
          N:     레이어 개수
        """
        super(Encoder, self).__init__()
        # 입력으로 받은 layer를 N번 복제
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # 최종 출력에 적용할 LayerNorm
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          x:    (batch_size, seq_len, d_model)
          mask: 패딩 마스크
        Returns:
          (batch_size, seq_len, d_model) — 인코더 최종 출력
        """
        # 순서대로 모든 레이어 통과
        for layer in self.layers:
            x = layer(x, mask)
        # 마지막에 정규화
        return self.norm(x)
