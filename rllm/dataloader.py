# rllm/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import numpy as np
import logging
from typing import Union, Optional, Dict, List, Any, Iterator

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """
    ▷ 현대적인 LLM 방식의 데이터셋 구현
    ▷ Huggingface 토크나이저를 사용하여 토큰화 및 전처리 수행
    ▷ 슬라이딩 윈도우 방식으로 긴 텍스트를 고정 길이 청크로 분할

    주요 기능:
      1. Huggingface 토크나이저를 사용한 토큰화
      2. max_length를 초과하는 경우 stride 간격으로 겹치게 청크 분할
      3. 패딩 및 어텐션 마스크 지원
      4. 캐싱 시스템으로 반복 접근 최적화
      5. 인과적 언어 모델링(CLM)을 위한 입력/레이블 생성
    """
    def __init__(self,
                 dataset,
                 max_length: int = 1024,
                 stride: int = 512,
                 column_name: str = '0',
                 tokenizer_name: str = "google/gemma-3-1b-pt",
                 padding: str = "max_length",
                 truncation: bool = True):
        """
        Args:
          dataset       : HuggingFace Dataset 또는 유사한 형태의 sequence 객체
          max_length    : 최대 시퀀스 길이
          stride        : 슬라이딩 윈도우 보폭
          column_name   : 데이터셋에서 사용할 컬럼 이름
          tokenizer_name: Huggingface 토크나이저 이름 또는 경로
          padding       : 패딩 방식 ('max_length' 또는 'longest')
          truncation    : 최대 길이 초과 시 잘라낼지 여부
        """
        # Huggingface 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # special token이 없는 경우 추가
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 데이터셋 및 설정값 저장
        self.dataset = dataset
        self.max_length = max_length
        self.stride = stride
        self.column_name = column_name
        self.padding = padding
        self.truncation = truncation
        
        # 입출력 캐싱을 위한 설정
        self.cache = {}
        self.cache_size = 100
        
        # 총 문서 수 (iterable 데이터셋은 -1로 설정)
        try:
            self.doc_count = len(dataset)
            self.is_iterable = False
        except (TypeError, ValueError):
            # streaming 데이터셋은 길이를 알 수 없음
            self.doc_count = -1
            self.is_iterable = True
            logger.info("Streaming 데이터셋 감지됨: 정확한 길이를 알 수 없습니다.")

        # iterable이 아닌 데이터셋인 경우에만 총 청크 수 계산 
        if not self.is_iterable:
            self._compute_total_chunks()
    
    def _compute_total_chunks(self):
        """
        전체 문서에서 생성될 수 있는 총 청크 수를 계산합니다.
        """
        if self.doc_count == 0:
            self.total_chunks = 0
            return
        
        # 첫 문서로 샘플 계산
        sample_text = self.dataset[0][self.column_name]
        # 토큰화
        tokens = self.tokenizer.encode(sample_text, add_special_tokens=False)
        
        # 문서당 생성될 청크 수 계산
        chunks_per_doc = max(1, (len(tokens) - self.max_length) // self.stride + 1)
        
        # 전체 문서에 대한 총 청크 수
        self.total_chunks = chunks_per_doc * self.doc_count
    
    def __len__(self):
        """총 청크 수 반환"""
        # streaming 데이터셋은 길이를 알 수 없으므로, 임의의 큰 값 반환
        if self.is_iterable:
            return int(1e9)  # 매우 큰 값 반환
        return self.total_chunks
    
    def _process_document(self, text: str):
        """
        단일 문서에서 모든 청크를 생성합니다.
        
        Args:
          text: 문서 텍스트
          
        Returns:
          List[Dict]: 각 청크별 입력 및 레이블 정보를 포함한 딕셔너리 리스트
        """
        # 텍스트가 없는 경우 빈 리스트 반환
        if text is None or len(text.strip()) == 0:
            return []
        
        # 토큰화 (특수 토큰 추가 X)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        
        # 슬라이딩 윈도우로 청크 생성
        for start in range(0, max(1, len(tokens) - self.max_length + 1), self.stride):
            # 입력 청크 (최대 max_length-1 길이, 마지막에 EOS 토큰 추가용 공간 확보)
            current_slice = tokens[start:start + self.max_length - 1]
            
            # 입력이 비어있거나 너무 짧으면 패딩된 청크 1개 생성
            if len(current_slice) < 2:  # 최소 2개 토큰 필요 (입력 1개 + 예측 1개)
                if len(tokens) > 0:  # 토큰이 있는 경우만
                    # 최소 1개 청크 생성하여 반환
                    current_slice = tokens[:min(len(tokens), self.max_length - 1)]
                else:
                    # 완전히 빈 경우 (토큰 없음) - 빈 청크 1개 생성
                    current_slice = [self.tokenizer.bos_token_id]
            
            # EOS 토큰 추가 (없는 경우)
            if self.tokenizer.eos_token_id not in current_slice:
                current_slice.append(self.tokenizer.eos_token_id)
            
            # 인과적 언어 모델링을 위한 입력과 레이블 생성
            # 입력: [토큰0, 토큰1, ..., 토큰N-1]
            # 레이블: [토큰1, 토큰2, ..., 토큰N]
            input_ids = current_slice[:-1]
            labels = current_slice[1:]
            
            # 어텐션 마스크 생성 (입력 토큰 부분만 1, 패딩은 0)
            attention_mask = [1] * len(input_ids)
            
            # 패딩 처리
            padding_length = self.max_length - 1 - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length  # -100은 loss 계산에서 무시됨
                attention_mask = attention_mask + [0] * padding_length
            
            # 결과 딕셔너리 생성
            chunk = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
            
            chunks.append(chunk)
            
            # 청크가 max_length에 딱 맞는 경우 더 이상 처리하지 않음
            if len(current_slice) < self.max_length:
                break
        
        # 청크가 없는 경우 빈 청크 하나 생성
        if not chunks:
            # 빈 청크 하나 생성
            empty_input = [self.tokenizer.bos_token_id]
            empty_label = [self.tokenizer.eos_token_id]
            
            # 패딩 처리
            padding_length = self.max_length - 1 - len(empty_input)
            input_ids = empty_input + [self.tokenizer.pad_token_id] * padding_length
            labels = empty_label + [-100] * padding_length
            attention_mask = [1] * len(empty_input) + [0] * padding_length
            
            chunks.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            })
        
        return chunks
    
    def __getitem__(self, idx: int):
        """
        idx에 해당하는 청크를 반환합니다.
        
        Args:
          idx: 청크 인덱스
          
        Returns:
          Dict: 입력 ID, 레이블, 어텐션 마스크를 포함한 딕셔너리
        """
        # 캐시 확인
        if idx in self.cache:
            return self.cache[idx]
        
        # 문서 인덱스 계산
        doc_idx = idx % self.doc_count if self.doc_count > 0 else idx
        
        try:
            # 해당 문서의 텍스트 가져오기
            text = self.dataset[doc_idx][self.column_name]
            
            # 해당 문서의 모든 청크 가져오기
            chunks = self._process_document(text)
            
            # 청크가 없는 경우 기본 청크 생성
            if not chunks:
                return self._create_default_chunk()
            
            # 청크 인덱스 계산
            chunk_idx = 0 if self.doc_count <= 0 else (idx // self.doc_count) % len(chunks)
            
            # 해당 청크 반환
            item = chunks[chunk_idx]
            
            # 캐시에 저장
            if len(self.cache) < self.cache_size:
                self.cache[idx] = item
            
            return item
        except Exception as e:
            logger.warning(f"데이터 처리 중 오류 발생: {e}. 기본 청크를 생성합니다.")
            return self._create_default_chunk()
    
    def _create_default_chunk(self):
        """기본 빈 청크 생성"""
        empty_input = [self.tokenizer.bos_token_id]
        empty_label = [self.tokenizer.eos_token_id]
        
        # 패딩 처리
        padding_length = self.max_length - 1 - len(empty_input)
        input_ids = empty_input + [self.tokenizer.pad_token_id] * padding_length
        labels = empty_label + [-100] * padding_length
        attention_mask = [1] * len(empty_input) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def collate_fn(self, batch):
        """
        배치 데이터를 모아서 처리하는 커스텀 collate 함수
        
        Args:
          batch: 샘플 리스트
          
        Returns:
          Dict: 배치 텐서를 포함한 딕셔너리
        """
        # 각 키별로 텐서 스택
        result = {
            key: torch.stack([sample[key] for sample in batch])
            for key in batch[0].keys()
        }
        
        return result


class StreamingDataset(IterableDataset):
    """
    ▷ Huggingface 스트리밍 데이터셋을 처리하기 위한 IterableDataset
    ▷ 메모리 효율적으로 대용량 데이터셋을 처리
    """
    def __init__(self,
                 dataset,
                 max_length: int = 1024,
                 stride: int = 512,
                 column_name: str = 'text',
                 tokenizer_name: str = "google/gemma-3-1b-pt",
                 padding: str = "max_length",
                 truncation: bool = True,
                 buffer_size: int = 1000):
        """
        Args:
          dataset       : HuggingFace 스트리밍 Dataset
          max_length    : 최대 시퀀스 길이
          stride        : 슬라이딩 윈도우 보폭
          column_name   : 데이터셋에서 사용할 컬럼 이름
          tokenizer_name: Huggingface 토크나이저 이름 또는 경로
          padding       : 패딩 방식 ('max_length' 또는 'longest')
          truncation    : 최대 길이 초과 시 잘라낼지 여부
          buffer_size   : 셔플링을 위한 버퍼 크기
        """
        super().__init__()
        # Huggingface 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # special token이 없는 경우 추가
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 데이터셋 및 설정값 저장
        self.dataset = dataset
        self.max_length = max_length
        self.stride = stride
        self.column_name = column_name
        self.padding = padding
        self.truncation = truncation
        self.buffer_size = buffer_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """데이터셋을 순회하며 청크를 생성합니다."""
        buffer = []
        
        # 데이터셋에서 샘플을 가져와 버퍼에 추가
        for sample in self.dataset:
            if self.column_name not in sample:
                logger.warning(f"컬럼 '{self.column_name}'이 샘플에 없습니다: {list(sample.keys())}")
                continue
                
            text = sample[self.column_name]
            if not text or not isinstance(text, str):
                continue
                
            # 청크 생성
            chunks = self._process_document(text)
            buffer.extend(chunks)
            
            # 버퍼가 차면 셔플하고 반환
            if len(buffer) >= self.buffer_size:
                # 셔플
                np.random.shuffle(buffer)
                # 버퍼 내 청크 반환
                for chunk in buffer:
                    yield chunk
                # 버퍼 비우기
                buffer = []
        
        # 남은 버퍼 처리
        if buffer:
            np.random.shuffle(buffer)
            for chunk in buffer:
                yield chunk
    
    def _process_document(self, text: str) -> List[Dict[str, torch.Tensor]]:
        """단일 문서에서 청크를 생성합니다."""
        # 텍스트가 없는 경우 빈 리스트 반환
        if text is None or len(text.strip()) == 0:
            return []
        
        # 토큰화 (특수 토큰 추가 X)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        
        # 슬라이딩 윈도우로 청크 생성
        for start in range(0, max(1, len(tokens) - self.max_length + 1), self.stride):
            # 입력 청크 (최대 max_length-1 길이, 마지막에 EOS 토큰 추가용 공간 확보)
            current_slice = tokens[start:start + self.max_length - 1]
            
            # 너무 짧은 청크는 건너뜀
            if len(current_slice) < 4:  # 최소 길이 조정 가능
                continue
            
            # EOS 토큰 추가 (없는 경우)
            if self.tokenizer.eos_token_id not in current_slice:
                current_slice.append(self.tokenizer.eos_token_id)
            
            # 인과적 언어 모델링을 위한 입력과 레이블 생성
            input_ids = current_slice[:-1]
            labels = current_slice[1:]
            
            # 어텐션 마스크 생성
            attention_mask = [1] * len(input_ids)
            
            # 패딩 처리
            padding_length = self.max_length - 1 - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length  # -100은 loss 계산에서 무시됨
                attention_mask = attention_mask + [0] * padding_length
            
            # 결과 딕셔너리 생성
            chunk = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
            
            chunks.append(chunk)
        
        return chunks


def create_dataloader(dataset,
                      batch_size: int = 4,
                      max_length: int = 1024,
                      stride: int = 512,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      column_name: str = '0',
                      tokenizer_name: str = "google/gemma-3-1b-pt"):
    """
    Huggingface 토크나이저를 사용한 데이터로더 생성 헬퍼 함수
    
    Args:
      dataset       : HF Dataset 또는 리스트 형태 데이터셋
      batch_size    : 배치 크기
      max_length    : 최대 시퀀스 길이
      stride        : 슬라이딩 윈도우 보폭
      shuffle       : 셔플 여부
      num_workers   : DataLoader 병렬 워커 수
      column_name   : 데이터셋 컬럼 이름
      tokenizer_name: Huggingface 토크나이저 이름
      
    Returns:
      loader, tokenizer: DataLoader와 Tokenizer 인스턴스
    """
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 스트리밍 데이터셋인지 확인
    is_streaming = hasattr(dataset, '_ex_iterable') or isinstance(dataset, IterableDataset)
    
    if is_streaming:
        logger.info("스트리밍 데이터셋 감지됨: StreamingDataset으로 처리합니다.")
        # IterableDataset 생성
        ds = StreamingDataset(
            dataset=dataset,
            max_length=max_length,
            stride=stride,
            column_name=column_name,
            tokenizer_name=tokenizer_name,
            buffer_size=batch_size * 100,  # 셔플 버퍼 크기
        )
        # IterableDataset은 샘플링에 worker_init_fn이 필요하지 않음
        worker_init_fn = None
        # 스트리밍 데이터셋은 DataLoader에서 shuffle 매개변수를 사용하지 않음
        shuffle = False
    else:
        logger.info("일반 데이터셋 감지됨: CustomDataset으로 처리합니다.")
        # 일반 Dataset 생성
        ds = CustomDataset(
            dataset=dataset,
            max_length=max_length,
            stride=stride,
            column_name=column_name,
            tokenizer_name=tokenizer_name,
        )
        # worker_init_fn 필요 없음
        worker_init_fn = None
    
    # DataLoader 생성
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and not is_streaming,  # 스트리밍일 때는 False
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=ds.collate_fn if hasattr(ds, 'collate_fn') else None,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,  # 워커가 있을 때만 지속적 워커 사용
    )
    
    return loader, tokenizer
