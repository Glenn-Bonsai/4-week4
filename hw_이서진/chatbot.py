import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 페이지 설정
st.set_page_config(
    page_title="사람인 채용 FAQ 챗봇",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 사람인 채용 FAQ 챗봇")
st.caption("Powered by GPT-4o + ChromaDB")
st.divider()


# ChromaDB 연결
@st.cache_resource
def load_collection():
    db_client  = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection("saramin_collection")
    return collection

collection = load_collection()

# RAG 함수 정의
def get_embedding(text: str, model: str = "text-embedding-3-large") -> list:
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def retrieve(query: str, top_k: int = 3) -> dict:
    query_embedding = get_embedding(query)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )


def generate_answer(query: str, top_k: int = 3) -> str:
    results     = retrieve(query, top_k)
    found_docs  = results["documents"][0]
    found_metas = results["metadatas"][0]

    if not found_docs:
        return "관련 채용공고를 찾지 못했습니다. 다른 키워드로 질문해 주세요."

    # context 구성
    context_parts = []
    for doc_text, meta in zip(found_docs, found_metas):
        header = f"[{meta.get('기업명', '')} | {meta.get('공고명', '')}]"
        context_parts.append(f"{header}\n{doc_text}")
    context_str = "\n\n---\n\n".join(context_parts)

    system_prompt = """
당신은 사람인(Saramin) 채용공고 데이터를 기반으로 구직자의 질문에 답변하는
채용 FAQ 어시스턴트입니다. 다음 원칙을 반드시 지키세요.

1. 제공된 채용공고 문서에 근거해서만 답변하세요.
2. 문서에 없는 내용은 추측하거나 만들어내지 마세요.
   - 정보가 없을 경우 "해당 공고에는 관련 정보가 명시되어 있지 않습니다"라고 하세요.
3. 답변은 간결하고 구조적으로 작성하세요.
   - 여러 공고를 비교할 때는 공고명을 명확히 구분하세요.
4. 채용 URL이 있을 경우 답변 말미에 함께 안내하세요.
5. 사용자가 한국어로 질문하면 한국어로 답변하세요.
6. 친절하고 전문적인 어투를 유지하세요.
"""

    user_prompt = f"""아래는 검색된 채용공고 정보입니다:

{context_str}

질문: {query}

위 공고 정보를 바탕으로 질문에 답변해 주세요."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        stream=True,   # 스트리밍으로 답변을 실시간 출력
    )

    # 스트리밍 제너레이터 반환
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# 채팅 UI
# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! 사람인 채용공고 FAQ 챗봇입니다. 😊\n\n"
                "궁금한 채용 정보를 질문해 주세요.\n\n"
                "**예시 질문**\n"
                "- 신입도 지원 가능한 공고 알려줘\n"
                "- AI 관련 직무 자격요건이 어떻게 돼?\n"
                "- 정규직 채용 중인 회사 알려줘"
            ),
        }
    ]

# 대화 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if query := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가 & 표시
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 어시스턴트 답변 스트리밍
    with st.chat_message("assistant"):
        answer = st.write_stream(generate_answer(query, top_k=3))

    # 완성된 답변 기록 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})