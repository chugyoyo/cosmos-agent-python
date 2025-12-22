from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_engine import qa_chain

app = FastAPI(title="LoRA + RAG LLM API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        # LangChain 的调用方式
        response = qa_chain.invoke({"query": request.question})

        # 提取结果
        answer = response['result']
        # 提取来源文档及其元数据
        sources = [doc.page_content[:100] + "..." for doc in response['source_documents']]

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)