from vectorstore import VECTOR_STORE
from utils import set_default_service_context

from llama_index import VectorStoreIndex, get_response_synthesizer

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor


def query(query_str: str) -> str:

    set_default_service_context()

    index = VectorStoreIndex.from_vector_store(VECTOR_STORE)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=2,
        vector_store_query_mode="default", # default, hybrid, sparse, text search
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.8)
        ]
    )

    response = query_engine.query(query_str)
    
    return response


if __name__ == "__main__":
    query_str = input("Input your query:")
    print(query(query_str))