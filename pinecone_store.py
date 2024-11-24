import os
from dotenv import load_dotenv
import time
from tqdm import tqdm
from uuid import uuid4

from pinecone import Pinecone,ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader

# .env 파일에서 환경 변수 로드
load_dotenv()

# 필요한 환경 변수 불러오기
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API 키와 환경 변수 확인
if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_ENVIRONMENT 환경 변수를 설정해주세요.")

EMBEDDING_DIMENSION = 1536

def load_multiple_pdf_documents(pdf_paths):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False,
    )

    documents = []
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            
            # 각 페이지의 텍스트를 더 작은 청크로 분할
            chunks = text_splitter.split_text(text)
            
            # 각 청크에 대해 메타데이터 추가
            for i, chunk in enumerate(chunks):
                # 명시적으로 page_content 키 사용
                doc = Document(
                    page_content=chunk,  # 반드시 page_content 키 사용
                    metadata={
                        'source': pdf_path,
                        'page': page_num + 1,
                        'chunk': i + 1,
                    }
                )
                documents.append(doc)
    
    return documents

def create_embeddings_and_db(documents):
    # OpenAI 임베딩 설정
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    
    # Pinecone 초기화
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 인덱스 이름 설정
    index_name = "law-index"
    
    # 기존 인덱스가 있다면 삭제
    if index_name in pc.list_indexes():
        pc.delete_index(index_name)
        time.sleep(20)  # 인덱스가 완전히 삭제될 때까지 대기
    
    # 새 인덱스 생성 - us-east-1 리전 설정
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # us-east-1 리전으로 변경
        )
    )
    
    time.sleep(20)  # 인덱스가 준비될 때까지 대기
    
    # 인덱스 가져오기
    index = pc.Index(index_name)
    
    # 문서를 작은 배치로 나누어 처리
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size)):
        # 배치 크기만큼 문서 선택
        batch = documents[i:i + batch_size]
        
        # 각 문서에 대해 고유 ID 생성
        ids = [str(uuid4()) for _ in batch]
        
        # 임베딩 생성 및 저장
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        embeds = embeddings.embed_documents(texts)
        
        # Pinecone에 직접 업로드
        vectors = []
        for id, embed, text, metadata in zip(ids, embeds, texts, metadatas):
            vectors.append({
                'id': id,
                'values': embed,
                'metadata': {
                    'page_content': text,
                    'text': text,
                    **metadata
                }
            })
        
        index.upsert(vectors=vectors)
        time.sleep(1)  # API 속도 제한 방지
    
    # 벡터스토어 반환
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"  # metadata의 text 필드를 사용
    )

if __name__ == "__main__":
    # PDF 파일 목록 정의
    pdf_paths = ["1.pdf", "2.pdf", "3.pdf", "4.pdf"]
    
    print("PDF 문서 로딩 중...")
    documents = load_multiple_pdf_documents(pdf_paths)
    print(f"총 {len(documents)}개의 PDF 문서가 로드되었습니다.")
    
    print("임베딩 생성 및 Pinecone DB 저장 중...")
    create_embeddings_and_db(documents)