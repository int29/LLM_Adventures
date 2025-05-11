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


class RMSNorm(nn.Module):
    """
    ▷ Root Mean Square Layer Normalization
    ▷ 표준 LayerNorm과 달리 평균 제거(centering) 단계가 없음
    ▷ 계산이 더 효율적이며 특히 LLM에서 성능이 향상됨
    """
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        """
        Args:
          size: 정규화할 feature 차원 크기 (d_model)
          dim:  정규화를 적용할 차원 (기본값: 마지막 차원)
          eps:  수치 안정성을 위한 작은 상수
        """
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, seq_len, features) 형태의 입력
        Returns:
          정규화된 출력, 같은 shape
        """
        # x^2의 평균을 계산 (RMS)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        # x를 RMS로 나누어 정규화 (scaling 적용)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        # 학습 가능한 가중치 적용
        return self.weight * x_normed

    def reset_parameters(self):
        """가중치를 1로 초기화"""
        nn.init.ones_(self.weight)


# Apex의 FusedRMSNorm을 사용할 수 있는지 확인하고 가능하면 사용
try:
    import apex
    class FusedRMSNorm(apex.normalization.FusedRMSNorm):
        """
        ▷ Apex 라이브러리의 고속 CUDA 구현을 사용한 RMSNorm
        ▷ GPU 메모리 최적화 및 성능 향상
        """
        def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
            super().__init__(size, eps=eps, elementwise_affine=True)
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(size))
            self.dim = dim
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.ones_(self.weight)
    
    # FusedRMSNorm을 사용하는 경우 기본 RMSNorm 대신 사용
    NormLayer = FusedRMSNorm
except:
    # Apex를 사용할 수 없는 경우 기본 RMSNorm 사용
    NormLayer = RMSNorm


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
      1) 정규화(LN/RMSNorm) → 2) 서브레이어 적용 → 3) Dropout → 4) Residual 연결
    ▷ Pre-LN/RMSNorm 아키텍처 방식 (원래 논문의 Post-LN과 다름)
    """
    def __init__(self, size: int, dropout: float, use_rms_norm: bool = True):
        """
        Args:
          size:         d_model 크기
          dropout:      드롭아웃 비율
          use_rms_norm: RMSNorm 사용 여부 (False면 LayerNorm 사용)
        """
        super(SublayerConnection, self).__init__()
        # 정규화 레이어 설정 (RMSNorm 또는 LayerNorm)
        self.norm = NormLayer(size) if use_rms_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Args:
          x:        (batch_size, seq_len, d_model)
          sublayer: 정규화 이후 적용할 함수(예: self-attn or feed-forward)
        Returns:
          Residual + Dropout + SublayerNorm 결과
        """
        # 1) 입력을 정규화
        norm_x = self.norm(x)
        # 2) 정규화된 입력에 서브레이어 적용
        sub_out = sublayer(norm_x)
        # 3) 드롭아웃 적용 후 잔차 연결 (원본 x에 더함)
        return x + self.dropout(sub_out)


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
                 dropout: float,
                 use_rms_norm: bool = True):
        """
        Args:
          size:         d_model 크기
          self_attn:    멀티-헤드 셀프 어텐션 모듈
          feed_forward: 위치별 Feed-Forward 모듈
          dropout:      드롭아웃 비율
          use_rms_norm: RMSNorm 사용 여부 (False면 LayerNorm 사용)
        """
        super(EncoderLayer, self).__init__()
        # 두 개의 sublayer 연결: [self-attn, feed-forward]
        self.sublayer = nn.ModuleList([
            SublayerConnection(size, dropout, use_rms_norm) for _ in range(2)
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
    ▷ 마지막에 추가 LayerNorm/RMSNorm 적용
    """
    def __init__(self, layer: EncoderLayer, N: int, use_rms_norm: bool = True):
        """
        Args:
          layer: 한 개의 EncoderLayer 인스턴스 (나중에 복제됨)
          N:     레이어 개수
          use_rms_norm: RMSNorm 사용 여부 (False면 LayerNorm 사용)
        """
        super(Encoder, self).__init__()
        # 입력으로 받은 layer를 N번 복제
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # 최종 출력에 적용할 정규화 레이어
        self.norm = NormLayer(layer.size) if use_rms_norm else LayerNorm(layer.size)

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
