from azure.core.credentials import TokenCredential
from openai import OpenAI

from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.models.embedding import EmbeddingRequest, EmbeddingResponse
from rag_core.infrastructure.azure_identity import create_azure_ai_foundry_token_provider


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        endpoint: str,
        deployment: str,
        credential: TokenCredential,
    ) -> None:
        token_provider = create_azure_ai_foundry_token_provider(credential)
        self._client = OpenAI(base_url=endpoint, api_key=token_provider)
        self._deployment = deployment

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response = self._client.embeddings.create(
            model=self._deployment,
            input=request.texts,
        )
        embeddings = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        return EmbeddingResponse(embeddings=embeddings)
