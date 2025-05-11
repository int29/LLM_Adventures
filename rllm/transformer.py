# rllm/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from rllm.embedding import Embeddings, PositionalEncoding, RotaryPositionEmbedding
from rllm.mha import MultiHeadAttention
from rllm.encoder import EncoderLayer, Encoder, PositionwiseFeedForward
from rllm.decoder import DecoderLayer, Decoder

def subsequent_mask(size: int) -> torch.Tensor:
    """
    ▷ 디코더에서 '미래 토큰'을 보지 않도록 하는 subsequent mask 생성 함수
    ▷ 반환값 shape: (1, size, size)
       - 상삼각행렬 위쪽(대각선 위)은 False, 그 외는 True
       - 이후 (batch, 1, size, size) 로 확장하여 사용
    """
    attn_shape = (1, size, size)
    # np.triu(..., k=1): 대각선 바로 위부터 1인 행렬 생성
    subsequent = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 1 → False, 0 → True 로 바꿔 mask 역할 수행
    return torch.from_numpy(subsequent) == 0

class Generator(nn.Module):
    """
    ▷ 디코더 최종 출력을 어휘(vocab) 차원으로 변환하고 log-softmax를 취하는 모듈
    """
    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        # d_model → vocab_size 선형 변환
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, tgt_len, d_model) 형태의 디코더 출력
        Returns:
          (batch_size, tgt_len, vocab_size) 형태의 log 확률 분포
        """
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """
    ▷ 'Attention Is All You Need' 논문의 인코더-디코더 Transformer 구현체
    """
    def __init__(self,
                 src_vocab: int,
                 tgt_vocab: int,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_length: int = 5000,
                 use_rope: bool = True,
                 use_flash_attn: bool = True):
        """
        Args:
          src_vocab:      소스 어휘 크기
          tgt_vocab:      타겟 어휘 크기
          d_model:        임베딩 및 모델 차원
          d_ff:           Feed-Forward 내부 은닉층 차원
          num_heads:      어텐션 헤드 수
          num_layers:     인코더/디코더 레이어 수
          dropout:        드롭아웃 확률
          max_length:     포지셔널 인코딩 최대 길이
          use_rope:       RoPE(Rotary Position Embedding) 사용 여부
          use_flash_attn: Flash Attention 사용 여부
        """
        super(Transformer, self).__init__()
        
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn

        # 1) 임베딩 레이어 설정 (RoPE 사용 여부에 따라 다르게 구성)
        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab),
            # RoPE를 사용하는 경우 별도의 positional encoding을 추가하지 않음
            # (RoPE는 MultiHeadAttention 내부에서 직접 적용됨)
            RotaryPositionEmbedding(d_model, max_length) if use_rope else PositionalEncoding(d_model, dropout, max_length)
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model, tgt_vocab),
            RotaryPositionEmbedding(d_model, max_length) if use_rope else PositionalEncoding(d_model, dropout, max_length)
        )

        # 2) 인코더/디코더에서 재사용할 attention & feed-forward 모듈 생성
        c = copy.deepcopy
        # MultiHeadAttention 초기화 시 RoPE 및 Flash Attention 사용 설정 전달
        attn = MultiHeadAttention(
            d_model, 
            num_heads, 
            dropout, 
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            max_length=max_length
        )
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 3) 인코더 스택: N개의 EncoderLayer
        #    - 각 레이어는 self-attention + feed-forward 포함
        self.encoder = Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout),
            num_layers
        )

        # 4) 디코더 스택: N개의 DecoderLayer
        #    - 각 레이어는 masked self-attn + src-attn + feed-forward 포함
        self.decoder = Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),
            num_layers
        )

        # 5) 최종 결과를 vocab 차원으로 변환하는 Generator
        self.generator = Generator(d_model, tgt_vocab)

        # 6) 모든 파라미터에 Xavier uniform 초기화 (2차원 이상 텐서에만)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _get_position_ids(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 시퀀스에 대한 위치 ID 텐서 생성

        Args:
          x: (batch_size, seq_len) 형태의 입력 텐서

        Returns:
          (batch_size, seq_len) 형태의 위치 ID 텐서
        """
        batch_size, seq_len = x.size()
        # 기본 위치 ID: 0부터 seq_len-1까지 (배치별로 복제)
        position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return position_ids

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        ▷ 소스 문장을 인코더를 통해 처리
        Args:
          src:         (batch_size, src_len) 토큰 ID 텐서
          src_mask:    (batch_size, 1, 1, src_len) 소스 패딩 마스크
          position_ids: 위치 ID 텐서 (RoPE에서 사용)
        Returns:
          (batch_size, src_len, d_model) 형태의 인코더 최종 출력
        """
        # 1) 임베딩 + 포지셔널 인코딩
        # 2) EncoderLayer 스택 통과
        # RoPE 사용 시 position_ids를 생성하고 MultiHeadAttention에 전달
        if position_ids is None and self.use_rope:
            position_ids = self._get_position_ids(src)
            
        x = self.src_embed(src)
        for layer in self.encoder.layers:
            # RoPE를 사용하는 경우 position_ids 전달
            if self.use_rope:
                # self-attention에 position_ids 전달
                x = layer.sublayer[0](x, lambda x_norm: layer.self_attn(
                    x_norm, x_norm, x_norm, mask=src_mask, position_ids=position_ids
                ))
                # feed-forward 처리
                x = layer.sublayer[1](x, layer.feed_forward)
            else:
                # 기존 방식대로 처리
                x = layer(x, src_mask)
        
        return self.encoder.norm(x)

    def decode(self,
               memory: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor,
               tgt_position_ids: torch.Tensor = None,
               memory_position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        ▷ 디코더를 통해 타겟 문장을 생성 준비
        Args:
          memory:           인코더의 출력 (batch, src_len, d_model)
          src_mask:         소스 패딩 마스크
          tgt:              (batch_size, tgt_len) 타겟 토큰 ID 텐서
          tgt_mask:         (batch_size, 1, tgt_len, tgt_len) 타겟 패딩+subsequent 마스크
          tgt_position_ids: 타겟 시퀀스 위치 ID 텐서 (RoPE에서 사용)
          memory_position_ids: 메모리 시퀀스 위치 ID 텐서 (RoPE에서 사용)
        Returns:
          (batch_size, tgt_len, d_model) 형태의 디코더 출력
        """
        # RoPE 사용 시 없는 position_ids를 생성
        if self.use_rope:
            if tgt_position_ids is None:
                tgt_position_ids = self._get_position_ids(tgt)
                
        # 1) 임베딩 + 포지셔널 인코딩
        x = self.tgt_embed(tgt)
        
        # 2) DecoderLayer 스택 통과
        for layer in self.decoder.layers:
            if self.use_rope:
                # masked self-attention에 tgt_position_ids 전달
                x = layer.sublayer[0](x, lambda x_norm: layer.self_attn(
                    x_norm, x_norm, x_norm, mask=tgt_mask, position_ids=tgt_position_ids
                ))
                
                # encoder-decoder attention에 memory_position_ids 전달
                # (교차 어텐션에서는 일반적으로 RoPE를 적용하지 않을 수도 있지만, 여기서는 옵션으로 제공)
                x = layer.sublayer[1](x, lambda x_norm: layer.src_attn(
                    x_norm, memory, memory, mask=src_mask, position_ids=memory_position_ids
                ))
                
                # feed-forward 처리
                x = layer.sublayer[2](x, layer.feed_forward)
            else:
                # 기존 방식대로 처리
                x = layer(x, memory, src_mask, tgt_mask)
        
        return self.decoder.norm(x)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        ▷ Transformer 전체 순전파(F→E→D)를 수행
        Args:
          src:      소스 토큰 텐서
          tgt:      타겟 토큰 텐서
          src_mask: 소스 패딩 마스크
          tgt_mask: 타겟 패딩+미래 마스크
        Returns:
          (batch_size, tgt_len, d_model) 형태의 결과 텐서
        """
        # RoPE를 사용하는 경우 위치 ID 생성
        src_position_ids = self._get_position_ids(src) if self.use_rope else None
        tgt_position_ids = self._get_position_ids(tgt) if self.use_rope else None
        
        # 1) 인코더 실행
        enc_out = self.encode(src, src_mask, position_ids=src_position_ids)
        
        # 2) 디코더 실행
        dec_out = self.decode(enc_out, src_mask, tgt, tgt_mask, 
                              tgt_position_ids=tgt_position_ids, 
                              memory_position_ids=src_position_ids)
        
        return dec_out
