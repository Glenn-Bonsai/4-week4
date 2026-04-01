import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ChromaDB 초기화
def init_db(db_path: str = "./chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    try:
        dbclient.delete_collection(name="saramin_collection")
        print("기존 컬렉션 삭제 완료")
    except Exception:
        pass
    collection = dbclient.create_collection(name="saramin_collection")
    print("새 컬렉션 생성 완료")
    return dbclient, collection

# json 로드
def load_jobs(json_path: str) -> list:
    """
    사람인 채용공고 JSON 파일을 읽어 리스트로 반환합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"총 {len(data)}개 공고 로드 완료")
    return data

# dict → 텍스트 변환
def job_to_text(job: dict) -> str:
    parts = [
        f"[기업명] {job.get('기업명', '').strip()}",
        f"[공고명] {job.get('공고명', '').strip()}",
        f"[경력] {job.get('경력', '').strip()}",
        f"[학력] {job.get('학력', '').strip()}",
        f"[고용형태] {job.get('고용형태', '').strip()}",
        f"[근무지] {job.get('근무지', '').strip()}",
    ]

    if job.get("주요업무", "").strip():
        parts.append(f"[주요업무]\n{job['주요업무'].strip()}")
    if job.get("자격요건", "").strip():
        parts.append(f"[자격요건]\n{job['자격요건'].strip()}")
    if job.get("채용절차", "").strip():
        parts.append(f"[채용절차]\n{job['채용절차'].strip()}")
    if job.get("url", "").strip():
        parts.append(f"[공고 URL] {job['url'].strip()}")

    return "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

# 임베딩 생성
def get_embedding(text: str, model: str = "text-embedding-3-large") -> list:
    """
    OpenAI text-embedding-3-large 모델로 텍스트를 임베딩 벡터로 변환합니다.
    """
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# 메인: 로드 → 변환 → 청킹 → 임베딩 → DB 저장
if __name__ == "__main__":
    JSON_PATH = "./saramin_jobs.json"   # 파일 경로 필요 시 수정
    DB_PATH   = "./chroma_db"

    # DB 초기화
    dbclient, collection = init_db(DB_PATH)

    # 데이터 로드
    jobs = load_jobs(JSON_PATH)

    doc_id  = 0
    skipped = 0

    for job in jobs:
        full_text = job_to_text(job)

        # 기업명 + 공고명만 있고 내용이 거의 없는 공고는 제외
        if len(full_text.strip()) < 30:
            skipped += 1
            continue

        chunks = chunk_text(full_text, chunk_size=500, chunk_overlap=50)

        for idx, chunk in enumerate(chunks):
            doc_id += 1
            embedding = get_embedding(chunk)

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "기업명":      job.get("기업명", ""),
                    "공고명":      job.get("공고명", ""),
                    "경력":        job.get("경력", ""),
                    "학력":        job.get("학력", ""),
                    "고용형태":    job.get("고용형태", ""),
                    "근무지":      job.get("근무지", ""),
                    "url":         job.get("url", ""),
                    "chunk_index": idx,
                }],
                ids=[str(doc_id)],
            )
            print(f"  [{doc_id}] {job.get('기업명')} - {job.get('공고명')} (청크 {idx})")

    print(f"\n임베딩 완료: {doc_id}개 청크 저장 / {skipped}개 공고 스킵(내용 없음)")