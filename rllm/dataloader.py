# rllm/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class CustomDataset(Dataset):
    """
    ▷ 슬라이딩 윈도우(sliding window) 기법으로 긴 텍스트를
      고정 길이 토큰 청크(token chunk)로 잘라주는 Dataset

    주요 기능:
      1. 문서 단위로 토크나이즈
      2. 토큰 길이가 max_length 이상이면 stride 간격으로 겹치며 자름
      3. 너무 짧거나 비어 있으면 0으로 패딩된 최소 1개의 청크를 생성
      4. idx → (문서 인덱스, 청크 인덱스) 매핑
      5. 간단한 캐시(cache) 메커니즘으로 동일 조회를 빠르게 처리
    """
    def __init__(self,
                 dataset,
                 max_length: int = 1024,
                 stride: int = 512,
                 column_name: str = '0',
                 tokenizer: str = "gpt-4o"):
        """
        Args:
          dataset     : HuggingFace Dataset 혹은 비슷한 형태의 sequence 객체
          max_length  : 한 번에 읽어들일 토큰 최대 길이
          stride      : 청크를 겹칠 때, 다음 청크 시작 간격
          column_name : dataset의 어느 컬럼을 사용할지 (문자열)
          tokenizer   : tiktoken 모델 이름 (예: "gpt-4o")
        """
        # tiktoken 으로 토크나이저 인스턴스를 생성
        self.tokenizer = tiktoken.encoding_for_model(tokenizer)
        # 실제 원본 데이터 (리스트, Dataset 객체 등)
        self.dataset = dataset
        # 청크 관련 하이퍼파라미터 저장
        self.max_length = max_length
        self.stride = stride
        self.column_name = column_name

        # 자주 접근하는 idx → 결과를 캐싱하기 위한 dict
        # (idx: (input_tensor, target_tensor))
        self.cache = {}
        # 캐시에 최대 몇 개 청크를 저장할지
        self.cache_size = 100

        # dataset에 들어있는 총 문서 개수
        self.doc_count = len(dataset)

    def __len__(self):
        """
        전체 샘플(청크) 개수 리턴.
          - 한 문서에서 몇 개의 청크를 뽑아낼지를 계산
          - 전체 문서에 대해 곱해 최종 샘플 수 산출

        계산 방식:
          - 각 문서마다 (문서 길이 - max_length) // stride + 1 만큼 청크
          - 단, 문서가 너무 짧으면 최소 1개의 청크
          - 전체 문서에 대해 곱셈
        """
        if self.doc_count == 0:
            return 0

        # 예시 문서를 하나 가져와 토큰 길이 계산
        sample_text = self.dataset[0][self.column_name]
        sample_tokens = self.tokenizer.encode(sample_text)

        # (len(tokens) - max_length) 만큼 겹치며 잘라낼 때 생성될 청크 수
        # 문서가 짧으면 1이 되도록 max(1, ...)
        chunks_per_doc = max(1,
                             (len(sample_tokens) - self.max_length) // self.stride + 1)
        # 전체 문서 수 곱하기
        return chunks_per_doc * self.doc_count

    def _process_document(self, doc_idx: int):
        """
        단일 문서(doc_idx)에서 모든 청크를 생성해서 리스트로 반환.
        - 문서 하나를 토큰화하고
        - max_length 길이의 슬라이딩 윈도우 기반 청크를 생성
        - 청크가 없으면 패딩 청크 한 개라도 무조건 생성

        Returns:
          input_chunks  : List[LongTensor]  (각 청크별 입력 시퀀스)
          target_chunks : List[LongTensor]  (각 청크별 레이블 시퀀스, input 시프트 버전)
        """
        # 1) 문서 텍스트 획득 및 토큰화
        text = self.dataset[doc_idx][self.column_name]
        token_ids = self.tokenizer.encode(text)

        input_chunks = []
        target_chunks = []

        # 2) 슬라이딩 윈도우로 모든 가능한 청크 생성
        #    range는 0, stride, 2*stride, ... 에서 시작
        #    각 청크 길이는 max_length
        for start in range(0, max(1, len(token_ids) - self.max_length + 1), self.stride):
            # 입력용 청크: start ~ start+max_length
            inp = token_ids[start : start + self.max_length]
            # 레이블용 청크: 다음 단어로 시프트 (autoregressive 학습)
            tgt = token_ids[start + 1 : start + self.max_length + 1]

            # 길이가 정확히 max_length일 때만 사용
            if len(inp) == self.max_length and len(tgt) == self.max_length:
                input_chunks.append(torch.tensor(inp, dtype=torch.long))
                target_chunks.append(torch.tensor(tgt, dtype=torch.long))

        # 3) 만약 청크가 하나도 생성되지 않는 경우 (문서가 너무 짧거나 빈 경우)
        #    최소 1개의 패딩 청크를 생성하도록 보장
        if not input_chunks:
            # 토큰 리스트 뒤에 0(패딩)으로 채우고 자르기
            pad_ids = token_ids + [0] * max(0, self.max_length - len(token_ids))
            pad_ids = pad_ids[:self.max_length]
            input_chunks  = [torch.tensor(pad_ids, dtype=torch.long)]
            target_chunks = [torch.tensor(pad_ids, dtype=torch.long)]

        return input_chunks, target_chunks

    def __getitem__(self, idx: int):
        """
        DataLoader가 요청하는 idx-th 샘플(청크)을 반환.
        - idx → 문서 번호 doc_idx / 청크 번호 chunk_idx로 변환
        - _process_document 결과에서 해당 청크를 꺼내 반환
        - 속도 향상을 위해 최근 100개 결과를 self.cache에 저장


        1) 캐시 확인
        2) idx → (문서 인덱스, 청크 인덱스) 계산
        3) _process_document로 실제 청크 리스트 얻기
        4) (input, target) 튜플 반환
        """
        # 1) 캐시 히트
        if idx in self.cache:
            return self.cache[idx]

        # 2) 전체 문서 수로 나눈 나머지가 문서 인덱스
        doc_idx = idx % self.doc_count

        # 3) 해당 문서의 모든 청크 리스트 계산
        inp_chunks, tgt_chunks = self._process_document(doc_idx)

        # 4) 문서마다 청크 개수만큼 순환하며 청크 인덱스 선택
        chunk_idx = (idx // self.doc_count) % len(inp_chunks)

        # 선택된 청크 반환
        item = (inp_chunks[chunk_idx], tgt_chunks[chunk_idx])

        # 5) 캐시에 남길 공간이 있으면 저장
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item

        return item


def create_dataloader(dataset,
                      batch_size: int = 4,
                      max_length: int = 1024,
                      stride: int = 512,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      column_name: str = '0'):
    """
    CustomDataset과 DataLoader를 함께 생성해주는 헬퍼 함수.

    Args:
      dataset     : HF Dataset or list-like
      batch_size  : 배치 크기
      max_length  : 청크 최대 길이
      stride      : 슬라이딩 윈도우 겹침 간격
      shuffle     : 셔플 여부
      num_workers : DataLoader 워커 수
      column_name : dataset 컬럼 키

    Returns:
      loader, tokenizer
        - loader    : torch.utils.data.DataLoader
        - tokenizer : 내부에서 사용된 tiktoken 인스턴스
    """
    ds = CustomDataset(dataset,
                       max_length=max_length,
                       stride=stride,
                       column_name=column_name)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,      # 배치 크기에 딱 맞지 않아도 마지막 배치를 버리지 않음
    )
    return loader, ds.tokenizer
