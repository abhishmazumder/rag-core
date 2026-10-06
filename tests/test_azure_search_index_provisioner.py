from unittest.mock import Mock

from azure.core.credentials import TokenCredential
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SearchIndex,
)

from rag_core.infrastructure.vectorstores import index_provisioner
from rag_core.infrastructure.vectorstores.index_provisioner import (
    create_or_update_azure_search_index,
)


def test_provisioner_calls_create_or_update_and_closes_client(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    client_class = mocker.patch.object(index_provisioner, "SearchIndexClient")
    client = client_class.return_value
    index = SearchIndex(
        name="configured-index",
        fields=[SearchField(name="id", type=SearchFieldDataType.String, key=True)],
    )

    create_or_update_azure_search_index(
        endpoint="https://search.example.net",
        index=index,
        credential=credential,
    )

    client_class.assert_called_once_with(
        endpoint="https://search.example.net",
        credential=credential,
    )
    client.create_or_update_index.assert_called_once_with(index)
    client.close.assert_called_once_with()
