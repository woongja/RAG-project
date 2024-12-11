from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
import boto3
import sqlite3

# AWS 세션 설정
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
    # ConversationBufferMemory는 일반적으로 llm 인자를 받지 않으므로 단독 생성
    # 필요하다면 langchain 버전 확인
    memory = ConversationBufferMemory(return_messages=True)
    return memory


def cnvs_chain(input_text, memory):

    chain_data = bedrock_chatbot()
    cnvs_chain = ConversationChain(llm=chain_data, memory=memory, verbose=True)
    chat_reply = cnvs_chain.predict(input=input_text)
    return chat_reply
