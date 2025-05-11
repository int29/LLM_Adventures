# rllm/training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from rllm.transformer import subsequent_mask

class LabelSmoothing(nn.Module):
    """
    ▷ 레이블 스무딩(Label Smoothing) 구현
       - 정답 토큰에 너무 과도한 확신을 주지 않도록 출력 분포를 부드럽게 만듭니다.
       - KLDivLoss를 사용해 예측 분포와 스무딩된 실제 분포 간 거리를 계산합니다.
    """
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1):
        """
        Args:
          size:        어휘 수 (vocab_size)
          padding_idx: 패딩 토큰 인덱스 (무시할 토큰)
          smoothing:   스무딩 계수 (default=0.1)
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # 정답에 할당할 확률
        self.smoothing = smoothing         # 나머지 확률을 분산시킬 양
        self.size = size

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:      (batch*tgt_len, vocab_size) 형태의 log-probabilities
          target: (batch*tgt_len,) 형태의 실제 토큰 인덱스
        Returns:
          스칼라 형태의 KLDivLoss 값
        """
        # 분포 크기 일치 확인
        assert x.size(1) == self.size

        # 1) (batch*tgt_len, vocab_size) 크기의 스무딩 분포 초기화
        true_dist = x.data.clone().fill_(self.smoothing / (self.size - 2))
        # 2) 실제 정답 인덱스 위치에 confidence 할당
        true_dist.scatter_(1, target.unsqueeze(1).data, self.confidence)
        # 3) 패딩 인덱스 위치는 모두 0
        true_dist[:, self.padding_idx] = 0
        # 4) 타깃이 패딩인 행 전체를 0으로 설정
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # KLDivLoss 계산 (log-probabilities vs. true_dist)
        return self.criterion(x, true_dist.detach())


class NoamOpt:
    """
    ▷ 'Attention Is All You Need' 논문의 Noam 스케줄러 구현
       - 학습 초기에는 learning rate를 점차 올렸다가,
         warmup 이후에는 스텝 수의 제곱근에 비례해 감소시킵니다.
    """
    def __init__(self, model_size: int, factor: float, warmup: int, optimizer):
        """
        Args:
          model_size: 모델 차원 (d_model)
          factor:     스케일링 계수
          warmup:     워밍업 단계 수
          optimizer:  실제 내부 옵티마이저 (예: Adam)
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        """내부 옵티마이저의 기울기 초기화 호출"""
        self.optimizer.zero_grad()

    def step(self):
        """
        1) 스텝 카운트 증가
        2) 현재 learning rate 계산 및 적용
        3) 옵티마이저 스텝 수행
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        """
        학습률 공식:
          lr = factor * (model_size^-0.5 * min(step^-0.5, step * warmup^-1.5))
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** -0.5 *
            min(step ** -0.5, step * self.warmup ** -1.5)
        )


def create_masks(src: torch.Tensor, tgt: torch.Tensor = None):
    """
    ▷ 소스/타깃 패딩 마스크 및 subsequent 마스크 생성
    Args:
      src: (batch, src_len) 형태의 소스 토큰 ID
      tgt: (batch, tgt_len) 형태의 타깃 토큰 ID (없으면 None)
    Returns:
      src_mask: (batch, 1, 1, src_len) 형태의 소스 패딩 마스크
      tgt_mask: (batch, 1, tgt_len, tgt_len) 형태의 결합 마스크 (있을 때만)
    """
    # 1) 소스 패딩 마스크: 토큰이 0(패딩)이 아닌 위치만 True
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch,1,1,src_len]

    if tgt is None:
        return src_mask

    # 2) 타깃 패딩 마스크
    tgt_pad = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch,1,1,tgt_len]
    # 3) 타깃 subsequent 마스크: (1,tgt_len,tgt_len) → (1,1,tgt_len,tgt_len)
    tgt_sub = subsequent_mask(tgt.size(1)).to(tgt.device).unsqueeze(1)
    # 4) 둘을 AND 결합 → [batch,1,tgt_len,tgt_len]
    tgt_mask = tgt_pad & tgt_sub

    return src_mask, tgt_mask


def train_epoch(model, dataloader, criterion, optimizer, device, log_interval: int = 100):
    """
    ▷ 한 epoch 동안 학습을 수행하고, 평균 token loss를 반환
    Args:
      model:      Transformer 또는 DDP 래퍼 모델
      dataloader: DataLoader (input, target) 배치를 제공
      criterion:  LabelSmoothing 인스턴스
      optimizer:  NoamOpt 래퍼
      device:     'cuda' 또는 'cpu'
      log_interval: 몇 배치마다 로그를 출력할지
    Returns:
      epoch_loss: token당 평균 loss
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    # DDP 래퍼면 실제 모델을 꺼내기
    real_model = model.module if hasattr(model, 'module') else model

    for i, (src, tgt_out) in enumerate(dataloader):
        # 1) 입력을 디바이스로 이동
        src = src.to(device)
        tgt_out = tgt_out.to(device)
        # 디코더 입력은 시프트 버전 (여기선 편의상 같은 src 사용)
        tgt_in = src

        # 2) 마스크 생성
        src_mask, tgt_mask = create_masks(src, tgt_in)
        # 3) 순전파
        out = model(src, tgt_in, src_mask, tgt_mask)
        # 4) generator 및 loss 계산
        logits = real_model.generator(out)  # [batch, tgt_len, vocab_size]
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            tgt_out.contiguous().view(-1)
        )

        # 5) 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6) loss와 토큰 수 집계
        n_tokens = (tgt_out != 0).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

        # 7) 중간 로그 출력
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start
            print(f"Batch {i+1}/{len(dataloader)} — "
                  f"Loss/token: {total_loss/total_tokens:.4f}, "
                  f"Tokens/sec: {total_tokens/elapsed:.2f}")
            start = time.time()
            total_loss = 0.0
            total_tokens = 0

    return total_loss / max(1, total_tokens)


def evaluate(model, dataloader, criterion, device):
    """
    ▷ 검증 데이터로 모델 평가를 수행
    Args:
      model:     Transformer 또는 DDP 래퍼 모델
      dataloader: DataLoader
      criterion: LabelSmoothing 인스턴스
      device:    'cuda' 또는 'cpu'
    Returns:
      eval_loss: token당 평균 loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    real_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for src, tgt_out in dataloader:
            src = src.to(device)
            tgt_out = tgt_out.to(device)
            tgt_in = src

            src_mask, tgt_mask = create_masks(src, tgt_in)
            out = model(src, tgt_in, src_mask, tgt_mask)
            logits = real_model.generator(out)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.contiguous().view(-1)
            )

            n_tokens = (tgt_out != 0).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens

    return total_loss / max(1, total_tokens)
