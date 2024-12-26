from loguru import logger

from llm_twin.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_twin.model.inference.run import InferenceExecutor
from llm_twin.settings import settings

if __name__ == "__main__":
    query = "Write me a post about AWS SageMaker inference endpoints."
    logger.info(f"Running inference for text: '{query}'")

    llm_inference = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
    )

    inference_executor = InferenceExecutor(llm_inference, query)

    answer = inference_executor.execute()
    logger.info(f"Answer: '{answer}'")
