# backend_1.py
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader
import boto3
import glob
from datetime import datetime
import json
import os
import torch

torch.cuda.empty_cache()

# AWS 세션 설정 (Bedrock)
session = boto3.Session(profile_name="default", region_name="ap-northeast-2")
bedrock_client = session.client("bedrock-runtime", region_name="ap-northeast-2")

def bedrock_chatbot():
    bedrock_llm = BedrockChat(
        client=bedrock_client,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={"temperature": 0.5, "top_p": 1}
    )
    return bedrock_llm

def buff_memory():
    # ConversationBufferMemory는 단순히 대화 기록을 보관.
    memory = ConversationBufferMemory(return_messages=True)
    return memory

# 메타데이터 파일 경로
METADATA_FILE = "processed_data.json"

def load_processed_docs():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

def save_processed_docs(processed_docs):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=4)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={'device': 'cuda:1'}  # CPU에서 실행
)

# 벡터 스토어가 이미 있으면 로드, 없으면 새로 생성
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

def add_new_information(text: str):
    if not os.path.exists("data"):
        os.makedirs("data")
    filename = f"latest_info_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
    file_path = os.path.join("data", filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    new_docs = [Document(page_content=text)]
    vectorstore.add_documents(new_docs)
    vectorstore.persist()

    # 메타데이터에 추가
    processed_docs = load_processed_docs()
    processed_docs.append(filename)
    save_processed_docs(processed_docs)

# 텍스트 분할을 위한 TextSplitter 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 각 Chunk의 최대 길이 (문자 수)
    chunk_overlap=30  # Chunk 간 겹치는 부분
)


# 앱 시작 시 data 폴더 내 아직 처리되지 않은 파일만 임베딩
def process_new_data():
    processed_docs = load_processed_docs()
    new_docs = []

    # data 폴더에서 TXT와 PDF 파일 처리
    for file_path in glob.glob("data/*"):
        filename = os.path.basename(file_path)
        if filename in processed_docs:
            continue  # 이미 처리된 파일 건너뜀

        print(f"Processing file: {filename}")
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            print(f"Unsupported file format: {file_path}")
            continue

        # 텍스트 분할
        split_documents = text_splitter.split_documents(docs)
        chunk_count = len(split_documents)
        print(f"File '{filename}' split into {chunk_count} chunks.")

        if chunk_count > 0:
            new_docs.extend(split_documents)
            processed_docs.append(filename)
    
    # 새 문서를 벡터 스토어에 추가 및 저장
    if new_docs:
        vectorstore.add_documents(new_docs)
        vectorstore.persist()
        save_processed_docs(processed_docs)
        print(f"Processed and saved {len(new_docs)} new chunks.")

# 전체 문서 수 가져오기
total_docs = vectorstore._collection.count()

# 실행 시 새로운 문서만 추가
process_new_data()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": total_docs
    }
)

def cnvs_chain(input_text, memory):
    # 1. 벡터 스토어에서 문서 검색
    docs = retriever.get_relevant_documents(input_text) if retriever else []

    if docs:
        # 각 문서를 구분하여 컨텍스트 생성
        context = "\n\n".join(
            [f"Document {i+1}:\n{d.page_content}" for i, d in enumerate(docs)]
        )
        prompt = f"""제공된 문서를 우선적으로 참고하되, 문서에 없는 내용에 대해서는 당신이 알고 있는 다른 지식을 활용하여 질문에 답변해주세요.

        문서:
        {context}

        질문: {input_text}"""
    else:
        # 검색 결과가 없을 경우 기본 프롬프트 생성
        prompt = f"질문: {input_text}"

    # LLM 호출 및 응답 생성
    chain_data = bedrock_chatbot()
    cnvs_chain = ConversationChain(llm=chain_data, memory=memory, verbose=True)
    chat_reply = cnvs_chain.predict(input=prompt)
    return chat_reply