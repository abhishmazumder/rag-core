from azure.core.credentials import TokenCredential
from openai import OpenAI

from rag_core.domain.interfaces.response_model import ResponseModel
from rag_core.domain.models.response import ResponseRequest, ResponseResponse
from rag_core.infrastructure.azure_identity import create_azure_ai_foundry_token_provider


class OpenAIResponseModel(ResponseModel):
    def __init__(
        self,
        endpoint: str,
        deployment: str,
        credential: TokenCredential,
    ) -> None:
        token_provider = create_azure_ai_foundry_token_provider(credential)
        self._client = OpenAI(base_url=endpoint, api_key=token_provider)
        self._deployment = deployment

    def generate(self, request: ResponseRequest) -> ResponseResponse:
        response = self._client.responses.create(
            model=self._deployment,
            input=[
                {
                    "type": "message",
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
        )
        return ResponseResponse(content=response.output_text)
