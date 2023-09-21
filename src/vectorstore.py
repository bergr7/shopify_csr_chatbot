import os
import weaviate
from llama_index.vector_stores import WeaviateVectorStore

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# weaviate client
WEAVIATE_CLIENT = weaviate.Client(
    url = "http://localhost:8080",
    additional_headers = {
        "X-OpenAI-Api-Key": OPENAI_API_KEY
    }
)

VECTOR_STORE = WeaviateVectorStore(
    weaviate_client=WEAVIATE_CLIENT,
    index_name="ShopifyHowToGuides",
)