from rag_core.config import Settings
from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.infrastructure import create_azure_credential
from rag_core.infrastructure.models.embedding.openai import OpenAIEmbeddingModel


def create_embedding_model(settings: Settings) -> EmbeddingModel:
    endpoint = settings.azure_ai_foundry_embedding_endpoint
    deployment = settings.azure_ai_foundry_embedding_model
    if not endpoint or not deployment:
        raise ValueError(
            "Azure AI Foundry embedding endpoint and model/deployment must be configured."
        )

    return OpenAIEmbeddingModel(
        endpoint=endpoint,
        deployment=deployment,
        credential=create_azure_credential(),
    )
