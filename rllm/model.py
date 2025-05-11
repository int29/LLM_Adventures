# rllm/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from typing import Dict, Tuple, Any, Optional

from rllm.transformer import Transformer, subsequent_mask
from rllm.training import LabelSmoothing, NoamOpt


class TransformerLightningModule(pl.LightningModule):
    """
    ▷ PyTorch Lightning을 활용한 Transformer 모델 구현
    ▷ 학습, 검증, 테스트 로직을 Lightning 프레임워크에 맞춰 통합
    ▷ RoPE(Rotary Position Embedding) 및 Flash Attention 지원
    ▷ Huggingface 토크나이저 통합 및 현대적 LLM 학습 방식 지원
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
                 pad_idx: int = 0,
                 lr_factor: float = 2.0,
                 warmup_steps: int = 4000,
                 label_smoothing: float = 0.1,
                 use_rope: bool = True,
                 use_flash_attn: bool = True,
                 tokenizer = None):
        """
        Args:
          src_vocab:       소스 어휘 크기
          tgt_vocab:       타겟 어휘 크기
          d_model:         임베딩 및 모델 차원
          d_ff:            Feed-Forward 내부 은닉층 차원
          num_heads:       어텐션 헤드 수
          num_layers:      인코더/디코더 레이어 수
          dropout:         드롭아웃 확률
          max_length:      포지셔널 인코딩 최대 길이
          pad_idx:         패딩 토큰 인덱스
          lr_factor:       학습률 스케일링 계수
          warmup_steps:    웜업 스텝 수 
          label_smoothing: 레이블 스무딩 계수
          use_rope:        RoPE(Rotary Position Embedding) 사용 여부
          use_flash_attn:  Flash Attention 사용 여부
          tokenizer:       Huggingface 토크나이저 (None 가능)
        """
        super(TransformerLightningModule, self).__init__()
        
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer
        
        # 기본 Transformer 모델 생성
        self.model = Transformer(
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_length,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn
        )
        
        # 손실 함수: 레이블 스무딩 적용
        self.criterion = LabelSmoothing(
            size=tgt_vocab, 
            padding_idx=pad_idx, 
            smoothing=label_smoothing
        )
        
        # 학습 관련 하이퍼파라미터 저장
        self.d_model = d_model
        self.lr_factor = lr_factor
        self.warmup_steps = warmup_steps
        self.pad_idx = pad_idx
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn

    def create_masks(self, 
                    input_ids: torch.Tensor, 
                    attention_mask: Optional[torch.Tensor] = None,
                    is_causal: bool = True):
        """
        입력 시퀀스에 대한 마스크 생성
        
        Args:
            input_ids: 입력 토큰 ID (batch_size, seq_len)
            attention_mask: 어텐션 마스크 (batch_size, seq_len) or None
            is_causal: 인과적 마스크 사용 여부
            
        Returns:
            (batch_size, 1, 1/seq_len, seq_len) 형태의 마스크
        """
        batch_size, seq_len = input_ids.size()
        
        # 어텐션 마스크가 제공되지 않은 경우 패딩 마스크 생성
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_idx).long()
        
        # 패딩 마스크: (batch, 1, 1, seq_len)
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 인과적 마스크 적용 (필요한 경우)
        if is_causal:
            # subsequent 마스크: (1, seq_len, seq_len)
            causal_mask = subsequent_mask(seq_len).to(input_ids.device)
            # 최종 마스크: 패딩 마스크 + 인과적 마스크
            mask = padding_mask & causal_mask
            return mask
        else:
            return padding_mask
    
    def forward(self, 
               input_ids: torch.Tensor, 
               labels: Optional[torch.Tensor] = None,
               attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
          input_ids: (batch_size, seq_len) 입력 토큰 ID
          labels: (batch_size, seq_len) 레이블 토큰 ID
          attention_mask: (batch_size, seq_len) 어텐션 마스크
          
        Returns:
          예측 로그 확률 또는 손실
        """
        # 토큰이 1개만 있는 경우 (추론 단계를 위해)
        if input_ids.size(1) == 1:
            mask = self.create_masks(input_ids, attention_mask, is_causal=False)
            
            # RoPE에서 사용할 위치 ID 생성 (필요한 경우)
            position_ids = self.model._get_position_ids(input_ids) if self.use_rope else None
            
            # 인코더 실행
            memory = self.model.encode(input_ids, mask, position_ids=position_ids)
            
            # 출력 벡터와 확률 계산
            output = memory
            logits = self.model.generator(output)
            
            return logits
        
        # 일반적인 경우 (시퀀스가 2개 이상의 토큰)
        # 인코더-디코더 모드가 아닌 디코더 전용 모드로 실행
        # 입력 ID를 1개 시프트하여 디코더 입력 / 타깃 생성
        src = input_ids
        tgt_in = input_ids  # 디코더 전용 모델에서는 입력을 그대로 사용
        
        # 마스크 생성 (인과적 마스크)
        mask = self.create_masks(src, attention_mask, is_causal=True)
        
        # RoPE에서 사용할 위치 ID 생성 (필요한 경우)
        position_ids = self.model._get_position_ids(src) if self.use_rope else None
        
        # Transformer 포워드 패스 (인코더만 사용)
        # 이 구현에서는 인코더와 디코더를 같이 사용하지만 실제로는 디코더만 필요
        output = self.model.encode(src, mask, position_ids=position_ids)
        
        # 로그 확률 계산
        logits = self.model.generator(output)
        
        # 레이블이 제공된 경우 손실 계산
        if labels is not None:
            # 패딩된 위치 마스킹 (loss 계산 시 무시)
            # -100 값은 CrossEntropyLoss에서 무시됨
            mask = (labels != -100)
            
            # 손실 계산
            shift_logits = logits[..., :-1, :].contiguous()  # 마지막 토큰 제외
            shift_labels = labels[..., 1:].contiguous()  # 첫 토큰 제외
            
            # 패딩 마스킹
            shift_mask = mask[..., 1:].contiguous()
            
            # 손실 계산 (크로스 엔트로피)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            return {'loss': loss, 'logits': logits}
        
        return logits
    
    def _common_step(self, batch, batch_idx, step_type):
        """
        학습/검증/테스트에 공통적으로 사용되는 단계 처리
        
        Args:
          batch: 입력 배치 (딕셔너리 형태)
          batch_idx: 배치 인덱스
          step_type: 'train', 'val', 'test' 중 하나
          
        Returns:
          loss: 스칼라 손실값
        """
        # 배치에서 입력, 레이블, 어텐션 마스크 추출
        input_ids = batch['input_ids']
        labels = batch.get('labels', None)
        attention_mask = batch.get('attention_mask', None)
        
        # 모델 실행
        outputs = self.forward(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
        
        # 손실 가져오기
        loss = outputs['loss'] if isinstance(outputs, dict) else None
        
        # 로깅
        if loss is not None:
            self.log(f"{step_type}_loss", loss, prog_bar=True, 
                    on_step=(step_type=='train'), on_epoch=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """훈련 단계"""
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        """검증 단계"""
        return self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        """테스트 단계"""
        return self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        """
        Noam 학습률 스케줄링이 적용된 Adam 옵티마이저 설정
        """
        # 기본 Adam 옵티마이저 생성
        base_optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=0.0,  # Noam 스케줄러가 실제 lr을 동적으로 설정
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        
        # Lightning에서 LambdaLR을 사용하여 Noam 스케줄러 구현
        def noam_lambda(step):
            # Noam 스케줄링: factor * d_model^(-0.5) * min(step^(-0.5), step*warmup^(-1.5))
            step = max(1, step)  # 0으로 나누기 방지
            return self.lr_factor * (self.d_model ** -0.5) * min(
                step ** -0.5, 
                step * (self.warmup_steps ** -1.5)
            )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(base_optimizer, noam_lambda),
            'name': 'noam_schedule',
            'interval': 'step',  # step 단위로 스케줄링
            'frequency': 1
        }
        
        return [base_optimizer], [scheduler]
    
    def generate(self, 
                input_ids: torch.Tensor, 
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: int = 0,
                top_p: float = 0.9,
                do_sample: bool = True,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None):
        """
        자동 회귀 방식으로 텍스트 생성
        
        Args:
            input_ids: 입력 토큰 ID (batch_size, seq_len)
            max_length: 생성할 최대 토큰 수
            temperature: 샘플링 온도 (낮을수록 확정적)
            top_k: 샘플링할 상위 k개 토큰 (0이면 사용 안 함)
            top_p: 누적 확률 p까지 샘플링 (1.0이면 사용 안 함)
            do_sample: 샘플링 사용 여부 (False면 greedy decoding)
            pad_token_id: 패딩 토큰 ID
            eos_token_id: 종료 토큰 ID
            
        Returns:
            생성된 토큰 ID (batch_size, out_len)
        """
        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id
            
        if pad_token_id is None:
            pad_token_id = self.pad_idx
        if eos_token_id is None:
            eos_token_id = self.pad_idx
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # 생성된 시퀀스 (초기값은 입력)
        generated_ids = input_ids.clone()
        
        # 시퀀스가 완료되었는지 표시하는 플래그
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 자동 회귀 생성
        while generated_ids.shape[1] < max_length and not is_finished.all():
            # 가장 최근 입력으로 다음 토큰 예측
            outputs = self.forward(generated_ids, attention_mask=None)
            
            # 마지막 토큰의 logits만 사용
            next_token_logits = outputs[:, -1, :]
            
            # 온도 적용
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 샘플링
            if do_sample:
                # Top-k 샘플링
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) 샘플링
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 임계값보다 큰 누적 확률을 가진 토큰 제거
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 첫 번째 토큰은 항상 유지
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 원래 인덱스에 맞게 변환하여 마스크 적용
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 확률 분포에서 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # EOS 토큰이 생성된 배치 위치 표시
            is_finished = is_finished | (next_tokens == eos_token_id)
            
            # 생성된 토큰 추가
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 모든 배치가 EOS 생성했으면 중단
            if is_finished.all():
                break
        
        return generated_ids
    
    def predict_step(self, batch, batch_idx, max_length: int = 100):
        """
        추론 단계에서의 자동 회귀식 생성
        
        Args:
            batch: 입력 배치
            batch_idx: 배치 인덱스
            max_length: 최대 생성 길이
        Returns:
            생성된 시퀀스 텐서
        """
        # 입력 추출
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
        else:
            input_ids = batch
            attention_mask = None
        
        # 텍스트 생성
        return self.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        ) 