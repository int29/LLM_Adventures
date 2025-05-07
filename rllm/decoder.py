# rllm/decoder.py

import torch
import torch.nn as nn
import copy

from rllm.mha import MultiHeadAttention
from rllm.encoder import LayerNorm, SublayerConnection, PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Transformer 디코더의 한 레이어를 구현합니다.
    하나의 레이어는 세 개의 서브레이어로 구성됩니다:
      1) Masked Self-Attention
      2) Encoder-Decoder Attention
      3) Position-wise Feed-Forward
    각 서브레이어 앞뒤로 LayerNorm, Dropout, Residual 연결이 적용됩니다.
    """
    def __init__(self,
                 size: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float):
        """
        Args:
          size:         모델 차원 (d_model)
          self_attn:    디코더 자기 자신에 대한 어텐션 (masked self-attn)
          src_attn:     인코더 출력에 대한 어텐션 (encoder-decoder attn)
          feed_forward: 위치별 FFN 모듈
          dropout:      드롭아웃 비율
        """
        super(DecoderLayer, self).__init__()
        # LayerNorm + Dropout + Residual 을 수행하는 헬퍼
        # 총 3개의 sublayer를 위해 리스트로 만듭니다.
        self.sublayer = nn.ModuleList([
            SublayerConnection(size, dropout),  # 0: masked self-attn
            SublayerConnection(size, dropout),  # 1: encoder-decoder attn
            SublayerConnection(size, dropout)   # 2: feed-forward
        ])
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:        (batch_size, tgt_len, d_model) 형태의 디코더 입력
          memory:   (batch_size, src_len, d_model) 형태의 인코더 출력
          src_mask: (batch_size, 1, 1, src_len) 형태의 소스 패딩 마스크
          tgt_mask: (batch_size, 1, tgt_len, tgt_len) 형태의 타깃 마스크
        Returns:
          (batch_size, tgt_len, d_model) 형태의 출력 텐서
        """
        # 1) Masked Self-Attention 서브레이어
        #    이전 출력 x에 대해 자기 자신과의 어텐션 수행
        #    이후 Dropout → Residual 연결
        x = self.sublayer[0](
            x,
            lambda x_norm: self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        )

        # 2) Encoder-Decoder Attention 서브레이어
        #    x를 쿼리로, 인코더 출력을 키/값으로 하여 어텐션 수행
        x = self.sublayer[1](
            x,
            lambda x_norm: self.src_attn(x_norm, memory, memory, src_mask)
        )

        # 3) Position-wise Feed-Forward 서브레이어
        x = self.sublayer[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    N개의 DecoderLayer를 연결하고 마지막에 LayerNorm을 추가한 디코더 블록.
    """
    def __init__(self, layer: DecoderLayer, N: int):
        """
        Args:
          layer: 미리 정의된 DecoderLayer 인스턴스 (복제용)
          N:     레이어 개수
        """
        super(Decoder, self).__init__()
        # 동일한 구조의 레이어를 N개 복제
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # 최종 출력에 적용할 LayerNorm
        self.norm = LayerNorm(layer.size)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:        (batch_size, tgt_len, d_model) 형태의 임베딩 + 포지셔널 인코딩된 타깃 시퀀스
          memory:   (batch_size, src_len, d_model) 형태의 인코더 출력
          src_mask: (batch_size, 1, 1, src_len) 형태의 소스 패딩 마스크
          tgt_mask: (batch_size, 1, tgt_len, tgt_len) 형태의 타깃 마스크
        Returns:
          (batch_size, tgt_len, d_model) 형태의 디코더 최종 출력
        """
        # 각 레이어를 순차적으로 통과
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 마지막에 정규화
        return self.norm(x)
