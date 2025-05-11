#!/usr/bin/env python
# rllm/train_korean.py

import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy
import wandb
from datasets import load_dataset

from rllm.dataloader import create_dataloader
from rllm.model import TransformerLightningModule


def train_korean_llm(
    dataset_name="maywell/korean_textbooks",
    data_files="mmlu_high_school_statistics/train-00000-of-00001.parquet",
    output_dir="./korean_llm_output",
    model_name_or_path="google/gemma-3-1b-pt",
    batch_size=8,
    max_epochs=3,
    num_workers=4,
    d_model=768,
    d_ff=3072,
    num_heads=12,
    num_layers=12,
    dropout=0.1,
    max_length=1024,
    stride=512,
    lr_factor=1.0,
    warmup_steps=1000,
    streaming=True,
    precision="bf16-mixed",
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
    save_every_n_steps=100,
    project_name="korean-llm-training",
    run_name=None,
    seed=42,
    use_rope=True,
    use_flash_attn=True,
    gpus=1,
):
    """
    한국어 교과서 데이터를 사용하여 LLM 모델을 학습합니다.
    
    Args:
        dataset_name: Huggingface 데이터셋 이름
        data_files: 데이터셋 내 특정 파일 경로
        output_dir: 체크포인트 및 로그 저장 디렉토리
        model_name_or_path: 토크나이저 경로 (모델은 처음부터 학습)
        batch_size: 배치 크기
        max_epochs: 최대 에포크 수
        num_workers: 데이터 로더 워커 수
        d_model: 모델 차원
        d_ff: 피드포워드 레이어 차원
        num_heads: 어텐션 헤드 수
        num_layers: 레이어 수
        dropout: 드롭아웃 비율
        max_length: 최대 시퀀스 길이
        stride: 슬라이딩 윈도우 보폭
        lr_factor: 학습률 스케일링 계수
        warmup_steps: 학습률 웜업 스텝 수
        streaming: 데이터 스트리밍 모드 사용 여부
        precision: 연산 정밀도
        accumulate_grad_batches: 그래디언트 누적 배치 수
        gradient_clip_val: 그래디언트 클리핑 임계값
        save_every_n_steps: 몇 스텝마다 체크포인트 저장할지
        project_name: wandb 프로젝트 이름
        run_name: wandb 실행 이름 (None이면 자동 생성)
        seed: 랜덤 시드
        use_rope: RoPE 사용 여부
        use_flash_attn: Flash Attention 사용 여부
        gpus: 사용할 GPU 수
    """
    # 시드 고정
    pl.seed_everything(seed)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 데이터셋 로드 (streaming 모드)
    print(f"Loading dataset {dataset_name} in streaming mode")
    ds_train = load_dataset(
        dataset_name,
        data_files=data_files,
        split="train",
        streaming=streaming,
    )
    
    # 토크나이저 및 데이터 로더 생성
    train_loader, tokenizer = create_dataloader(
        dataset=ds_train,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
        column_name="text",  # 데이터셋의 텍스트 컬럼 이름 (필요시 조정)
        tokenizer_name=model_name_or_path,
    )
    
    # 토크나이저에서 vocab_size 가져오기
    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.pad_token_id
    
    # 모델 초기화
    model = TransformerLightningModule(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_length=max_length,
        pad_idx=pad_idx,
        lr_factor=lr_factor,
        warmup_steps=warmup_steps,
        label_smoothing=0.1,
        use_rope=use_rope,
        use_flash_attn=use_flash_attn,
        tokenizer=tokenizer,
    )
    
    # wandb 로거 설정
    if run_name is None:
        run_name = f"korean-llm-{d_model}-{num_layers}l-{max_length}ctx"
    
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        log_model="all",  # 모든 체크포인트 저장
    )
    
    # 중요 하이퍼파라미터 로깅
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
    
    # 체크포인트 및 LR 모니터링 콜백 설정
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{step}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=save_every_n_steps,
        save_last=True,
    )
    
    # 훈련 전략 설정 (DeepSpeed Zero-2 사용)
    if gpus > 1:
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            offload_parameters=False,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        )
    else:
        strategy = "auto"
    
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
    
    # 훈련 시작
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader)
    
    # wandb 종료
    wandb.finish()
    
    return model, tokenizer, trainer


if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description="한국어 LLM 모델 학습")
    
    # 데이터 관련 인자
    parser.add_argument("--dataset_name", type=str, default="maywell/korean_textbooks",
                      help="Huggingface 데이터셋 이름")
    parser.add_argument("--data_files", type=str, 
                      default="mmlu_high_school_statistics/train-00000-of-00001.parquet",
                      help="데이터셋 내 특정 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./korean_llm_output",
                      help="체크포인트 및 로그 저장 디렉토리")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-1b-pt",
                      help="토크나이저 경로")
    
    # 학습 관련 인자
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--max_epochs", type=int, default=3, help="최대 에포크 수")
    parser.add_argument("--num_workers", type=int, default=4, help="데이터 로더 워커 수")
    parser.add_argument("--d_model", type=int, default=768, help="모델 차원")
    parser.add_argument("--d_ff", type=int, default=3072, help="피드포워드 레이어 차원")
    parser.add_argument("--num_heads", type=int, default=12, help="어텐션 헤드 수")
    parser.add_argument("--num_layers", type=int, default=12, help="레이어 수")
    parser.add_argument("--dropout", type=float, default=0.1, help="드롭아웃 비율")
    parser.add_argument("--max_length", type=int, default=1024, help="최대 시퀀스 길이")
    parser.add_argument("--stride", type=int, default=512, help="슬라이딩 윈도우 보폭")
    parser.add_argument("--lr_factor", type=float, default=1.0, help="학습률 스케일링 계수")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="학습률 웜업 스텝 수")
    parser.add_argument("--no_streaming", action="store_false", dest="streaming",
                      help="데이터 스트리밍 모드 비활성화")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="연산 정밀도")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, 
                      help="그래디언트 누적 배치 수")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, 
                      help="그래디언트 클리핑 임계값")
    parser.add_argument("--save_every_n_steps", type=int, default=100, 
                      help="몇 스텝마다 체크포인트 저장할지")
    
    # wandb 관련 인자
    parser.add_argument("--project_name", type=str, default="korean-llm-training",
                      help="wandb 프로젝트 이름")
    parser.add_argument("--run_name", type=str, default=None, help="wandb 실행 이름")
    
    # 기타 인자
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--use_rope", action="store_true", default=True, 
                      help="RoPE 사용 여부")
    parser.add_argument("--no_rope", action="store_false", dest="use_rope", 
                      help="RoPE 사용하지 않음")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, 
                      help="Flash Attention 사용 여부")
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn", 
                      help="Flash Attention 사용하지 않음")
    parser.add_argument("--gpus", type=int, default=1, help="사용할 GPU 수")
    
    args = parser.parse_args()
    
    # 함수 호출
    train_korean_llm(
        dataset_name=args.dataset_name,
        data_files=args.data_files,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_length=args.max_length,
        stride=args.stride,
        lr_factor=args.lr_factor,
        warmup_steps=args.warmup_steps,
        streaming=args.streaming,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        save_every_n_steps=args.save_every_n_steps,
        project_name=args.project_name,
        run_name=args.run_name,
        seed=args.seed,
        use_rope=args.use_rope,
        use_flash_attn=args.use_flash_attn,
        gpus=args.gpus,
    ) 