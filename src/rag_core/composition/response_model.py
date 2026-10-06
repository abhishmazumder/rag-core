from rag_core.config import Settings
from rag_core.domain.interfaces.response_model import ResponseModel
from rag_core.infrastructure import create_azure_credential
from rag_core.infrastructure.models.response.openai import OpenAIResponseModel


def create_response_model(settings: Settings) -> ResponseModel:
    endpoint = settings.azure_ai_foundry_response_endpoint
    deployment = settings.azure_ai_foundry_response_model
    if not endpoint or not deployment:
        raise ValueError(
            "Azure AI Foundry response endpoint and model/deployment must be configured."
        )

    return OpenAIResponseModel(
        endpoint=endpoint,
        deployment=deployment,
        credential=create_azure_credential(),
    )
