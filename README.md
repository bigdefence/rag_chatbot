# LLaMA-3-Korean 기반 RAG 챗봇 🤖🇰🇷

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B)](https://streamlit.io/)
[![라이선스](https://img.shields.io/badge/라이선스-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![버전](https://img.shields.io/badge/버전-1.3.0-brightgreen)](https://github.com/yourusername/rag-chatbot)

LLaMA-3-Korean-Bllossom-8B 모델을 활용한 강력한 검색 증강 생성(RAG) 챗봇입니다. 이 애플리케이션을 통해 사용자는 지식 베이스 문서를 업로드하고 해당 내용을 기반으로 지능적인 대화를 나눌 수 있습니다.

## 🌟 주요 기능

- 📚 다양한 문서 형식 지원 (CSV, PDF, TXT)
- 🧠 LangChain을 이용한 고급 텍스트 처리
- 🔍 FAISS를 활용한 효율적인 정보 검색
- 💬 SentenceTransformer를 통한 자연어 이해
- 🤖 LLaMA-3-Korean 모델을 이용한 지능적 응답 생성
- 🌐 다국어 지원 (한국어 중심)
- 🖥️ 사용자 친화적인 Streamlit 인터페이스

## 🛠️ 설치 방법

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/bigdefence/rag_chatbot.git
   cd rag_chatbot
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. Streamlit 앱을 실행합니다:
   ```
   streamlit run app.py
   ```

## 🚀 사용 방법

1. 사이드바를 이용해 지식 베이스 파일(CSV, PDF, 또는 TXT)을 업로드합니다.
2. 파일이 처리되고 벡터 저장소에 로드될 때까지 기다립니다.
3. 채팅 입력창에 질문을 입력하여 AI 어시스턴트와 대화를 시작합니다.
4. 업로드된 지식 베이스와 LLaMA-3-Korean 모델을 기반으로 한 지능적인 응답을 받습니다.

## 🧰 사용된 기술

- [Streamlit](https://streamlit.io/): 웹 애플리케이션 인터페이스 구현
- [PyTorch](https://pytorch.org/): 딥러닝 프레임워크
- [Transformers](https://huggingface.co/transformers/): LLaMA-3-Korean 모델 활용
- [SentenceTransformers](https://www.sbert.net/): 의미론적 유사도 검색
- [LangChain](https://langchain.com/): 문서 로딩 및 처리
- [FAISS](https://github.com/facebookresearch/faiss): 고밀도 벡터의 효율적인 유사도 검색 및 클러스터링

## 👨‍💻 개발자 정보

- **이름**: 정강빈
- **버전**: 1.0.0

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 글

- [MLP-KTLim](https://huggingface.co/MLP-KTLim): LLaMA-3-Korean-Bllossom-8B 모델 제공
- Hugging Face 팀: 놀라운 transformers 라이브러리 개발
- LangChain 커뮤니티: 강력한 문서 처리 도구 제공

## 🤝 기여하기

기여, 이슈 제기, 기능 요청을 환영합니다! [이슈 페이지](https://github.com/bigdefence/rag_chatbot/issues)를 확인해 주세요.

## 📬 연락처

정강빈 - bigdefence@naver.com

프로젝트 링크: [https://github.com/bigdefence/rag-chatbot](https://github.com/bigdefence/rag_chatbot)
