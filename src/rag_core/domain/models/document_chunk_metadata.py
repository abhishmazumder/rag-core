from pydantic import BaseModel


class DocumentChunkMetadata(BaseModel):
    source_document_id: str
    source_name: str
    document_type: str
    page_number: int | None = None
    source_url: str | None = None
    candidate_id: str
    document_id: str
