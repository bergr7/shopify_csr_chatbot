from llama_index import (
    Document,
    download_loader,
    VectorStoreIndex
)

from llama_index.storage.storage_context import StorageContext

from llama_index.schema import TextNode
from llama_index.data_structs import IndexDict

from llama_index.text_splitter import TokenTextSplitter

from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor
)

from llama_index import VectorStoreIndex

import constants
from vectorstore import VECTOR_STORE
from utils import set_default_service_context

from typing import List
import os


def load_documents() -> List[Document]:
    """
    Load md documents from data directory using default MarkdownReader in Llama Hub.
    """
    MarkdownReader = download_loader("MarkdownReader")

    loader = MarkdownReader()
    
    documents: List = []
    for doc_path in constants.DOCUMENT_PATHS[:1]: # ! DELETE - limited to 1 for debugging
        docs = loader.load_data(file=doc_path)[:3] # ! DELETE - limited to 3 pages for debugging
        # add source as metadata
        source = os.path.basename(doc_path)
        for doc in docs:
            doc.metadata = {"source": source}
            doc.excluded_llm_metadata_keys = ["source"] # we do not want this metadata for the synthesis atm
            doc.excluded_embed_metadata_keys = ["source"] # we don't want to embed this data
        documents.extend(docs)
        
    return documents


def parse_documents(documents: List[Document]) -> List[TextNode]:
    """
    Parse documents into nodes and extract metadata with an LLM.
    """
    
    # split documents
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=256,
        chunk_overlap=20
    )
    
    metadata_extractor = MetadataExtractor(
        extractors=[
            TitleExtractor(nodes=5),
            SummaryExtractor(),
            KeywordExtractor(keywords=5),
        ]
    )
    
    # parser
    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=text_splitter,
        metadata_extractor=metadata_extractor,
        include_prev_next_rel=True,
        include_metadata=True,
    )
    
    nodes = node_parser.get_nodes_from_documents(documents=documents)
    
    return nodes


def construct_index(nodes: List[TextNode]) -> IndexDict:
    """
    Construct an index from nodes
    """
    storage_context = StorageContext.from_defaults(vector_store=VECTOR_STORE)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context) # * Embeddings are created when we create the index from nodes
    return index
    

def main() -> None:
    
    set_default_service_context()
    
    documents = load_documents()
    
    nodes = parse_documents(documents=documents)
    
    construct_index(nodes)


if __name__ == "__main__":
    main()