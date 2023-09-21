from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding


def set_default_service_context() -> None:
    """
    Set default global service context.
    """
    embed_model = OpenAIEmbedding() # default is text-embedding-ada-002
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=512)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    set_global_service_context(service_context)