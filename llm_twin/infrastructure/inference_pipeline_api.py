from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_twin.application.rag.retriever import ContextRetriever
from llm_twin.domain.embedded_chunks import EmbeddedChunk
from llm_twin.model.inference import LLMInferenceSagemakerEndpoint, InferenceExecutor
from llm_twin.settings import settings

app = FastAPI()


# Request structure for the FastAPI endpoints
class QueryRequest(BaseModel):
    query: str


# Response structure for the FastAPI endpoints
class QueryResponse(BaseModel):
    answer: str


def call_llm_service(query: str, context: str | None) -> str:
    # Call the SageMaker LLM microservice
    llm_inference = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
    )
    inference_executor = InferenceExecutor(
        llm_inference=llm_inference, query=query, context=context
    )
    answer = inference_executor.execute()

    return answer


def rag(query: str) -> str:
    """
    Define RAG business logic
    """
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query=query, k=4)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query=query, context=context)

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/healthcheck/")
def healthcheck():
    return "Health - OK"
