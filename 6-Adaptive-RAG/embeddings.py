# %%
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import json
from langchain.embeddings.base import Embeddings

# # 1. Create the RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# # Example long text to split
# text = """
# Bedrock is an AWS service that provides foundational models for various AI tasks.
# It includes text generation, image generation, and embedding models like Titan.
# LangChain is a framework that allows easy integration of these models for building language-based applications.
# """

# # 2. Split the text into chunks
# chunks = text_splitter.split_text(text)
# print(f"Chunks: {chunks}")

# # 3. Initialize Bedrock client with boto3
# client = boto3.client('bedrock-runtime', region_name='eu-central-1')

# # Function to embed a chunk of text
# def embed_text(chunk):
#     "{\"inputText\":\"this is where you place your input text\", \"dimensions\": 512, \"normalize\": true}"
#     response = client.invoke_model(
#         modelId='amazon.titan-embed-text-v2:0',
#         body=json.dumps({"inputText": chunk}),
#         contentType='application/json',
#     )
#     # Extract the embedding from the response
#     embedding = response['body'].read().decode('utf-8')  # Adjust based on the response format
#     return embedding

# # 4. Embed each chunk
# embeddings = [embed_text(chunk) for chunk in chunks]
# len(json.loads(embedding)['embedding'])
# Print or use the embeddings
# for i, embedding in enumerate(embeddings):
#     print(f"Embedding for chunk {i}: {embedding}")
#     break

# Step 1: Define the custom embedding class that calls Bedrock
class BedrockTitanEmbeddings(Embeddings):
    def __init__(self):
        # Initialize Bedrock client
        self.client = boto3.client('bedrock-runtime', region_name='eu-central-1')

    def embed(self, text):
        # Define the payload for embedding request
        body = {
            "inputText": text
        }

        # Call Bedrock Titan embedding model
        response = self.client.invoke_model(
            modelId='amazon.titan-embed-text-v1',  # Use the appropriate model ID
            contentType='application/json',
            body=json.dumps(body)
        )

        # Extract the embedding vector from the response
        embedding = json.loads(response['body'].read().decode('utf-8'))['embedding']
        return embedding

    def embed_documents(self, texts):
        # Embed a list of texts (used by Chroma)
        return [self.embed(text) for text in texts]
    
    def embed_query(self, query):
        # Embed a single query (used for query embedding)
        return self.embed(query)

# %%
