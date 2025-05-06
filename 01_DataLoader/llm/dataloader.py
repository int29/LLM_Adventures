import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken
from tqdm.notebook import tqdm  # 노트북 환경에서 진행률 표시

class CustomDataset(Dataset):
    def __init__(self, dataset, max_length=1024, stride=512, column_name='0', tokenizer="gpt-4o"):
        
        self.tokenizer = tiktoken.encoding_for_model(tokenizer)         # tiktoken 토크나이저 초기화
        self.dataset = dataset                                          # Hugging Face 데이터셋 저장
        self.max_length = max_length                                    # 최대 시퀀스 길이 저장
        self.stride = stride                                            # 슬라이딩 윈도우의 보폭(stride) 저장
        self.column_name = column_name                                  # 텍스트 데이터가 포함된 컬럼 이름 저장
        
        # 캐시 설정
        self.cache = {}                                                 # 간단한 캐시 딕셔너리
        self.cache_size = 100                                           # 8GB VRAM, 16GB RAM 환경에 맞게 설정
        self.doc_count = len(dataset)                                   # 전체 문서 수 및 인덱스 맵핑
    
    def __len__(self):
        # 첫 번째 문서를 기반으로 평균 청크 수 추정(Loader의 batch추정 및 진행률 위해서 __len__ 필요)
        if self.doc_count > 0:
            # 샘플 문서 토큰화
            sample_doc = self.dataset[0][self.column_name]
            sample_tokens = self.tokenizer.encode(sample_doc)
            
            # 가능한 청크 수 계산
            sample_chunks = max(0, (len(sample_tokens) - self.max_length) // self.stride + 1)
            
            # 전체 청크 수 추정
            return max(self.doc_count, sample_chunks * self.doc_count)
        
        return self.doc_count  # 기본값으로 문서 수 반환
    
    def _process_document(self, doc_idx):
    
        # 문서 텍스트 가져오기
        doc_text = self.dataset[doc_idx][self.column_name]
        
        # tiktoken으로 토큰화
        token_ids = self.tokenizer.encode(doc_text)
        
        input_chunks = []
        target_chunks = []
        
        # 슬라이딩 윈도우를 사용하여 토큰 ID 리스트를 청크로 나눔
        for i in range(0, len(token_ids) - self.max_length, self.stride):
            # 입력 청크: 인덱스 i부터 시작하는 max_length 길이의 토큰 시퀀스
            input_chunk = token_ids[i : i + self.max_length]
            
            # 타겟 청크: 입력보다 한 토큰 뒤에서 시작 (자동회귀 예측을 위한 1토큰 증분)
            # 예: 입력=[1,2,3,4], 타겟=[2,3,4,5]
            target_chunk = token_ids[i + 1 : i + self.max_length + 1]
            
            # 입력 청크와 타겟 청크의 길이가 모두 max_length인지 확인
            if len(input_chunk) == self.max_length and len(target_chunk) == self.max_length:
                # 텐서로 변환하여 리스트에 추가
                input_chunks.append(torch.tensor(input_chunk, dtype=torch.long))
                target_chunks.append(torch.tensor(target_chunk, dtype=torch.long))
        
        return input_chunks, target_chunks
    
    def __getitem__(self, idx):
        # 캐시 확인
        if idx in self.cache:
            return self.cache[idx]
        
        # 문서 인덱스 계산(입력된 idx를 문서 총 개수(self.doc_count)로 나눈 나머지를 사용합니다.)
        doc_idx = idx % self.doc_count
        
        # 문서 처리(고정슬라이드)
        input_chunks, target_chunks = self._process_document(doc_idx)
        
        # 처리된 청크가 없으면 다음 문서 시도
        if not input_chunks:
            return self.__getitem__((idx + 1) % len(self))
        
        # 청크 인덱스 계산
        # 청크로 문서를 나누기 때문에 마찬가지로 필요
        
        chunk_idx = (idx // self.doc_count) % max(1, len(input_chunks))

        # 결과 저장
        result = (input_chunks[chunk_idx % len(input_chunks)], target_chunks[chunk_idx % len(input_chunks)])
        
        # 캐시 업데이트 (제한된 크기)
        if len(self.cache) < self.cache_size:
            self.cache[idx] = result
        
        return result
    

    # 데이터 로딩 및 데이터로더 생성 함수
def create_dataloader(dataset, batch_size=4, max_length=1024, stride=512, shuffle=True, num_workers=0, column_name='0'):

    # CustomDataset 생성
    dataset = CustomDataset(
        dataset = dataset,
        max_length=max_length,
        stride=stride,
        column_name=column_name
    )
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if torch.cuda.is_available() else 0,  # CPU 전용 모드에서는 0으로 설정
        pin_memory=torch.cuda.is_available()  # GPU 사용 시 메모리 고정
    )
    
    return dataloader, dataset.tokenizer


# 메인 실행 코드 - 노트북 환경에서 바로 실행 가능
if __name__ == "__main__":
    # 이 코드는 Jupyter Notebook에서 직접 실행하거나
    # 아래와 같이 셀에서 따로 불러올 수 있습니다.
    
    # CUDA 사용 가능 여부 확인
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 하드웨어에 맞게 설정 조정
    batch_size = 4        # 8GB VRAM에 맞게 조정
    max_length = 1024     # 8GB VRAM에 맞게 조정
    stride = 512          # 절반 오버랩
    num_workers = 0       # 16GB RAM에 맞게 조정
    
    # 데이터로더 생성
    loader, tokenizer = create_dataloader(
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        num_workers=num_workers
    )
    
