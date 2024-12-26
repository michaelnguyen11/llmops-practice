import uvicorn

from llm_twin.infrastructure.inference_pipeline_api import app

if __name__ == "__main__":

    uvicorn.run("tools.ml_service:app", host="0.0.0.0", port=8000, reload=True)