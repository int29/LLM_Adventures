# rllm/trainer.py

import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.fabric import Fabric

from rllm.dataloader import create_dataloader
from rllm.model import TransformerLightningModule


def train_with_lightning(
    train_dataset,
    val_dataset=None,
    batch_size=32,
    max_epochs=10,
    num_workers=4,
    vocab_size=50000,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    max_length=1024,
    stride=512,
    lr_factor=2.0,
    warmup_steps=4000,
    label_smoothing=0.1,
    accumulate_grad_batches=1,
    precision="32-true",
    log_dir="./logs",
    ckpt_dir="./checkpoints",
    accelerator="auto",
    strategy="auto",
    devices="auto",
    column_name="0",
    pad_idx=0,
    use_rope=True,
    use_flash_attn=True,
    tokenizer_name="google/gemma-3-1b-pt",
):
    """
    ▷ PyTorch Lightning을 사용하여 Transformer 모델 학습

    Args:
        train_dataset: 학습 데이터셋
        val_dataset: 검증 데이터셋 (없으면 None)
        batch_size: 배치 크기
        max_epochs: 최대 에포크 수
        num_workers: 데이터 로더 워커 수
        vocab_size: 어휘 크기
        d_model: 모델 차원
        d_ff: 피드포워드 레이어 차원
        num_heads: 어텐션 헤드 수
        num_layers: 레이어 수
        dropout: 드롭아웃 비율
        max_length: 최대 시퀀스 길이
        stride: 슬라이딩 윈도우 보폭
        lr_factor: 학습률 스케일링 계수
        warmup_steps: 학습률 웜업 스텝 수
        label_smoothing: 레이블 스무딩 계수
        accumulate_grad_batches: 그래디언트 누적 배치 수
        precision: 연산 정밀도 (16, 32, "16-mixed", "bf16-mixed", "32-true")
        log_dir: 로그 디렉토리
        ckpt_dir: 체크포인트 디렉토리
        accelerator: 가속기 타입 ("cpu", "gpu", "tpu", "auto")
        strategy: 학습 전략 ("ddp", "deepspeed", "auto")
        devices: 사용할 디바이스 수
        column_name: 데이터셋 컬럼 이름
        pad_idx: 패딩 토큰 인덱스
        use_rope: RoPE(Rotary Position Embedding) 사용 여부
        use_flash_attn: Flash Attention 사용 여부
        tokenizer_name: Huggingface 토크나이저 이름
    """
    # 데이터 로더 생성
    train_loader, tokenizer = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
        column_name=column_name,
        tokenizer_name=tokenizer_name,
    )
    
    # 토크나이저에서 vocab_size 가져오기
    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.pad_token_id
    
    val_loader = None
    if val_dataset is not None:
        val_loader, _ = create_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=False,
            num_workers=num_workers,
            column_name=column_name,
            tokenizer_name=tokenizer_name,
        )
    
    # 모델 인스턴스 생성
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
        label_smoothing=label_smoothing,
        use_rope=use_rope,
        use_flash_attn=use_flash_attn,
        tokenizer=tokenizer,  # 토크나이저 전달 추가
    )
    
    # 로거 설정
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="transformer",
        version=None,  # 자동으로 버전 증가
    )
    
    # 체크포인트 콜백 설정
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="transformer-{epoch:02d}-{val_loss:.4f}" if val_loader else "transformer-{epoch:02d}",
        save_top_k=3,
        monitor="val_loss" if val_loader else None,
        mode="min" if val_loader else "max",
        save_last=True,
    )
    
    # 학습 프로그레스 바 설정
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    
    # Lightning 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar],
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
    )
    
    # 모델 학습
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    return model, trainer, tokenizer


def train_with_fabric(
    train_dataset,
    val_dataset=None,
    batch_size=32,
    max_epochs=10,
    num_workers=4,
    vocab_size=50000,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    max_length=1024,
    stride=512,
    lr_factor=2.0,
    warmup_steps=4000,
    label_smoothing=0.1,
    precision="32-true",
    accelerator="auto",
    devices="auto",
    column_name="0",
    pad_idx=0,
    ckpt_path=None,
    use_rope=True,
    use_flash_attn=True,
    tokenizer_name="google/gemma-3-1b-pt",
):
    """
    ▷ PyTorch Fabric을 사용하여 더욱 유연한 분산 학습 구현
    ▷ 단일 GPU에서도 동작하며, 멀티 GPU 환경으로 쉽게 확장 가능

    Fabric은 Lightning보다 더 낮은 추상화 수준에서 작동하여 
    분산 학습에 필요한 기본 기능만 제공합니다.
    """
    # Fabric 인스턴스 초기화
    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )
    fabric.launch()
    
    # 데이터 로더 생성
    train_loader, tokenizer = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
        column_name=column_name,
        tokenizer_name=tokenizer_name,
    )
    
    # 토크나이저에서 vocab_size 가져오기
    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.pad_token_id
    
    val_loader = None
    if val_dataset is not None:
        val_loader, _ = create_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=False,
            num_workers=num_workers,
            column_name=column_name,
            tokenizer_name=tokenizer_name,
        )
    
    # 모델 생성 및 초기화
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
        label_smoothing=label_smoothing,
        use_rope=use_rope,
        use_flash_attn=use_flash_attn,
        tokenizer=tokenizer,  # 토크나이저 전달 추가
    )
    
    # 옵티마이저 설정
    optimizer, scheduler = model.configure_optimizers()
    optimizer = optimizer[0]  # 리스트에서 첫 번째 옵티마이저만 사용
    scheduler = scheduler[0]["scheduler"]  # 첫 번째 스케줄러
    
    # 모델과 옵티마이저를 Fabric으로 설정
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    if val_loader:
        val_loader = fabric.setup_dataloaders(val_loader)
    
    # 체크포인트에서 불러오기 (있을 경우)
    if ckpt_path and os.path.exists(ckpt_path):
        fabric.print(f"체크포인트 {ckpt_path}에서 모델 불러오는 중...")
        state = fabric.load(ckpt_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        start_epoch = 0
        global_step = 0
    
    # 학습 루프
    for epoch in range(start_epoch, max_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        total_tokens = 0
        
        fabric.print(f"Epoch {epoch+1}/{max_epochs}")
        for batch_idx, batch in enumerate(train_loader):
            # 배치 처리 및 손실 계산
            loss = model.training_step(batch, batch_idx)
            
            # 역전파
            fabric.backward(loss)
            
            # 옵티마이저 스텝
            optimizer.step()
            optimizer.zero_grad()
            
            # 스케줄러 스텝
            scheduler.step()
            
            # 손실 누적
            train_loss += loss.item()
            # 효과적인 토큰 개수 (패딩 아닌 것) 계산
            mask = batch['attention_mask']
            n_tokens = mask.sum().item()
            total_tokens += n_tokens
            
            global_step += 1
            
            # 로깅 (일정 간격으로)
            if (batch_idx + 1) % 10 == 0:
                fabric.print(f"  Step {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.7f}")

        # 에포크 평균 손실 계산
        train_loss /= max(1, total_tokens)
        fabric.print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
        
        # 검증 루프 (있을 경우)
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_total_tokens = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    loss = model.validation_step(batch, batch_idx)
                    val_loss += loss.item()
                    # 토큰 개수 계산
                    mask = batch['attention_mask']
                    n_tokens = mask.sum().item()
                    val_total_tokens += n_tokens
            
            val_loss /= max(1, val_total_tokens)
            fabric.print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        # 체크포인트 저장
        save_path = os.path.join("checkpoints", f"transformer_epoch_{epoch+1}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fabric.save(
            save_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }
        )
        fabric.print(f"체크포인트 저장됨: {save_path}")
    
    return model, tokenizer


if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description="Transformer 모델 학습")
    parser.add_argument("--mode", type=str, choices=["lightning", "fabric"], default="fabric",
                      help="학습 모드 선택: lightning 또는 fabric")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--max_epochs", type=int, default=10, help="최대 에포크 수")
    parser.add_argument("--d_model", type=int, default=512, help="모델 차원")
    parser.add_argument("--num_heads", type=int, default=8, help="어텐션 헤드 수")
    parser.add_argument("--num_layers", type=int, default=6, help="레이어 수")
    parser.add_argument("--max_length", type=int, default=1024, help="최대 시퀀스 길이")
    parser.add_argument("--precision", type=str, default="32-true", help="연산 정밀도")
    parser.add_argument("--accelerator", type=str, default="auto", help="가속기 타입")
    parser.add_argument("--devices", type=int, default=1, help="사용할 디바이스 수")
    parser.add_argument("--ckpt_path", type=str, default=None, help="체크포인트 경로 (있을 경우)")
    parser.add_argument("--use_rope", action="store_true", default=True, help="RoPE(Rotary Position Embedding) 사용")
    parser.add_argument("--no_rope", action="store_false", dest="use_rope", help="RoPE 사용하지 않음")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Flash Attention 사용")
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn", help="Flash Attention 사용하지 않음")
    parser.add_argument("--tokenizer_name", type=str, default="google/gemma-3-1b-pt", help="Huggingface 토크나이저 이름")
    
    args = parser.parse_args()
    
    # 데이터셋 불러오기 (실제 사용 시 이 부분 수정 필요)
    # 예: HuggingFace의 datasets 라이브러리 사용
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    except:
        print("데이터셋을 불러올 수 없습니다. 테스트 데이터로 대체합니다.")
        # 테스트용 더미 데이터셋 생성
        train_dataset = [{"0": "This is a sample text for training."} for _ in range(100)]
        val_dataset = [{"0": "This is a sample text for validation."} for _ in range(20)]
    
    # 선택한 모드로 학습 실행
    if args.mode == "lightning":
        model, trainer, tokenizer = train_with_lightning(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_length=args.max_length,
            precision=args.precision,
            accelerator=args.accelerator,
            devices=args.devices,
            use_rope=args.use_rope,
            use_flash_attn=args.use_flash_attn,
            tokenizer_name=args.tokenizer_name,
        )
    else:  # fabric
        model, tokenizer = train_with_fabric(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_length=args.max_length,
            precision=args.precision,
            accelerator=args.accelerator,
            devices=args.devices,
            ckpt_path=args.ckpt_path,
            use_rope=args.use_rope,
            use_flash_attn=args.use_flash_attn,
            tokenizer_name=args.tokenizer_name,
        ) 