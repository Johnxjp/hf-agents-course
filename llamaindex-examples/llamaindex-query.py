import asyncio

import chromadb
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore

with open("hf_token.txt") as f:
    hf_token = f.read().strip()

inference_model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection(name="alfred_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
llm = HuggingFaceInferenceAPI(model_name=inference_model_name, token=hf_token, provider="auto")
query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        embed_model,
    ],
    vector_store=vector_store,
)

evaluator = FaithfulnessEvaluator(llm=llm)


def create_index():
    pipeline.run(documents=[Document.example()])


def query_index():
    response = query_engine.query("What is the meaning of life?")
    print(response)
    eval_result = evaluator.evaluate_response(response=response)
    print(eval_result.passing)


if __name__ == "__main__":
    create_index()
    query_index()
