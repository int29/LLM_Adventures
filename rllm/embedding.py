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
