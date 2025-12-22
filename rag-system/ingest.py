from langchain_community.document_loaders import CSVLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

VECTOR_DB_PATH = "../vector_store"
DATA_SOURCE = "../data/cleaned/rag_docs.csv"  # 假设这是清洗后的RAG知识库


def create_vector_db():
    # 1. 加载数据
    loader = CSVLoader(file_path=DATA_SOURCE, encoding='utf-8')
    documents = loader.load()

    # 2. 文本切分 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    # 3. 初始化 Embedding 模型 (使用开源模型，免费且效果不错)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    # 4. 存入 Chroma 向量数据库 (本地持久化)
    if os.path.exists(VECTOR_DB_PATH):
        # 如果存在则加载并追加，或者删除重建
        print("向量库已存在，正在追加数据...")

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print(f"向量库构建完成，共存入 {len(texts)} 个切片。")


if __name__ == "__main__":
    create_vector_db()
