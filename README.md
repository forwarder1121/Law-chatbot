# Law Chatbot

## 프로젝트 개요

Law Chatbot은 법률 관련 질문에 대한 답변을 제공하는 챗봇입니다. 이 챗봇은 PDF 문서에서 정보를 추출하고, OpenAI의 임베딩 모델을 사용하여 질문에 대한 적절한 답변을 생성합니다.

## 기능

-   PDF 문서에서 텍스트 추출
-   텍스트를 청크로 분할하여 저장
-   Pinecone을 사용한 벡터 저장소
-   OpenAI API를 통한 질문 응답

## 설치 방법

1. 이 리포지토리를 클론합니다.

    ```bash
    git clone https://github.com/forwarder1121/Law-chatbot.git
    cd Law-chatbot
    ```

2. 필요한 패키지를 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

3. `.env` 파일을 생성하고 다음 환경 변수를 설정합니다.
    ```
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENVIRONMENT=your_pinecone_environment
    OPENAI_API_KEY=your_openai_api_key
    ```

## 사용 방법

1. PDF 문서를 준비합니다. (예: `심리상담.pdf`)
2. 다음 명령어로 Pinecone 데이터베이스를 초기화합니다.

    ```bash
    python pinecone_store.py
    ```

3. Streamlit 앱을 실행합니다.

    ```bash
    streamlit run streamlit_app.py
    ```

4. 브라우저에서 `http://localhost:8501`에 접속하여 챗봇을 사용합니다.

## 기여

기여를 원하시는 분은 이 리포지토리를 포크한 후, 변경 사항을 커밋하고 Pull Request를 제출해 주세요.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
