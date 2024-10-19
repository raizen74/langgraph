from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from embeddings import BedrockTitanEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [
    item for sublist in docs for item in sublist
]  # each item in the sublist is the document that we want

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Code used to create the embeddings
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=BedrockTitanEmbeddings(),
#     persist_directory="./.chroma",
# )

# Perform similarity searches
retriever = Chroma(collection_name="rag-chroma", persist_directory="./.chroma", embedding_function=BedrockTitanEmbeddings()).as_retriever()
