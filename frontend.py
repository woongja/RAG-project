import streamlit as st
import backend as be  # backend.py의 함수들 가져오기

st.title("RAG CHATBOT")

# 세션 초기화
if "memory" not in st.session_state:
    st.session_state.memory = be.buff_memory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 채팅 기록 표시
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message["text"])

# 사용자 입력 처리
input_text = st.chat_input("질문을 입력하세요.")

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append({"role": "user", "text": input_text})

    # 백엔드에서 응답 생성
    chat_response = be.cnvs_chain(input_text=input_text, memory=st.session_state.memory)
    
    with st.chat_message("assistant"):
        st.markdown(chat_response)
    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})