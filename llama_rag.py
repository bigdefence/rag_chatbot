import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="RAG 챗봇 with LLaMA-3-Korean", layout="wide")

# 모델 로드 함수들
@st.cache_resource
def load_llm():
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 모델 로드
try:
    tokenizer, model = load_llm()
    sentence_model = load_sentence_model()
    cross_encoder = load_cross_encoder()
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# LangChain을 사용한 문서 로딩 및 처리
def process_file(uploaded_file):
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 파일 형식에 따라 적절한 로더 선택
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.type == "text/csv":
            loader = CSVLoader(tmp_file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")

        # 문서 로드
        documents = loader.load()

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 임베딩 모델 설정
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # Vectorstore 생성
        vectorstore = FAISS.from_documents(texts, embeddings)

        return vectorstore
    finally:
        # 임시 파일 삭제
        os.unlink(tmp_file_path)

# 관련 정보 검색
def retrieve_info(query, vectorstore, top_k=5):
    docs = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

def generate_response(query, retrieved_info):
    PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
    context = " ".join(retrieved_info)
    
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=500,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

# Streamlit 앱
st.title('RAG 챗봇 with LLaMA-3-Korean')

# 사이드바에 파일 업로더 추가
uploaded_file = st.sidebar.file_uploader("지식베이스 파일을 업로드하세요 (CSV, PDF, TXT)", type=["csv", "pdf", "txt"])

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file is not None:
    try:
        # 파일 처리
        with st.spinner('지식베이스를 처리 중입니다...'):
            st.session_state.vectorstore = process_file(uploaded_file)
        
        st.success('지식베이스가 성공적으로 로드되었습니다!')

    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        st.session_state.vectorstore = None

# 채팅 인터페이스
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("무엇을 도와드릴까요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is not None:
        with st.chat_message("assistant"):
            with st.spinner('답변을 생성 중입니다...'):
                retrieved_info = retrieve_info(prompt, st.session_state.vectorstore)
                response = generate_response(prompt, retrieved_info)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning('먼저 지식베이스 파일을 업로드해주세요.')

# 앱 실행 방법 안내
st.sidebar.markdown("""
## 사용 방법
1. 사이드바에서 CSV, PDF, 또는 TXT 파일을 업로드하세요.
2. 채팅 입력창에 질문을 입력하세요.
3. AI 어시스턴트의 응답을 확인하세요.
""")

# 앱 정보
st.sidebar.markdown("---")
st.sidebar.info("이 앱은 LLaMA-3-Bllossom-8B 모델을 사용한 RAG(Retrieval-Augmented Generation) 챗봇입니다.")
st.sidebar.info("개발자: 정강빈")
st.sidebar.info("버전: 1.3.0")