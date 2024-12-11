import streamlit as st
import backend as be  # backend.py에 구현된 query_bedrock_rag 함수

st.title("AWS Bedrock 기반 RAG 챗봇")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 채팅 기록 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# 사용자 입력 처리
input_text = st.chat_input("질문을 입력하세요.")

if input_text:
    # 사용자 입력 출력
    with st.chat_message("user"):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role": "user", "text": input_text})

    # Bedrock 기반 RAG 응답 생성
    chat_response = be.query_bedrock_rag(input_text)

    # 생성된 응답 출력
    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
