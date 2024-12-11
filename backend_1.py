# backend.py

from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import boto3
import glob
from datetime import datetime
import json
import os

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
METADATA_FILE = "processed_docs.json"

def load_processed_docs():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

def save_processed_docs(processed_docs):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=4)

# ---------------------
# 벡터 스토어 초기화
# ---------------------
# document_list는 사전에 로드한 문서 리스트라고 가정
# 예: document_list = [Document(page_content="최신 정보 텍스트1"), Document(page_content="최신 정보 텍스트2"), ...]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 벡터 스토어가 이미 있으면 로드, 없으면 새로 생성
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

def add_new_information(text: str):
    if not os.path.exists("docs"):
        os.makedirs("docs")
    filename = f"latest_info_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
    file_path = os.path.join("docs", filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    new_docs = [Document(page_content=text)]
    vectorstore.add_documents(new_docs)
    vectorstore.persist()

    # 메타데이터에 추가
    processed_docs = load_processed_docs()
    processed_docs.append(filename)
    save_processed_docs(processed_docs)

# 앱 시작 시 docs 폴더 내 아직 처리되지 않은 파일만 임베딩
def process_new_docs():
    processed_docs = load_processed_docs()
    new_docs = []
    for file_path in glob.glob("docs/*.txt"):
        filename = os.path.basename(file_path)
        if filename not in processed_docs:
            # 새로운 문서
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            new_docs.append(Document(page_content=content))
            processed_docs.append(filename)
    
    if new_docs:
        vectorstore.add_documents(new_docs)
        vectorstore.persist()
        save_processed_docs(processed_docs)

# 실행 시 새로운 문서만 추가
process_new_docs()
    
def cnvs_chain(input_text, memory):
    # 1. 벡터 스토어에서 문서 검색
    docs = retriever.get_relevant_documents(input_text) if retriever else []

    if docs:
        # RAG: 관련 문서가 있을 경우 해당 문서를 컨텍스트로 LLM 응답
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"제공된 문서를 우선적으로 참고하되, 문서에 없는 내용에 대해서는 당신이 알고 있는 다른 지식을 활용하여 질문에 답변해주세요.\n\n문서:\n{context}\n\n질문: {input_text}"
    else:
        # DB에도 없으면 기존 LLM 대화
        prompt = input_text
    chain_data = bedrock_chatbot()
    cnvs_chain = ConversationChain(llm=chain_data, memory=memory, verbose=True)
    chat_reply = cnvs_chain.predict(input=prompt)
    return chat_reply