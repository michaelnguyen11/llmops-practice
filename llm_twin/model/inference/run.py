from __future__ import annotations

from llm_twin.domain.inference import Inference
from llm_twin.settings import settings


class InferenceExecutor:
    def __init__(
        self,
        llm_inference: Inference,
        query: str,  # user input
        context: str | None = None,  # to do RAG
        prompt: str | None = None,  # customize prompt
    ) -> None:
        self.llm_inference = llm_inference
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
            You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
            User query: {query}
            Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        # format the prompt with the user’s query and context
        prompt_formatted = self.prompt.format(query=self.query, context=self.context)

        self.llm_inference.set_payload(
            inputs=prompt_formatted,
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,  # maximum number of new tokens the model is allowed to generate
                "repetition_penalty": 1.1,  # discourage the model from generating repetitive text
                "temperature": settings.TEMPERATURE_INFERENCE,  # control the randomness of the output
            },
        )

        answer = self.llm_inference.inference()[0]["generated_text"]

        return answer
