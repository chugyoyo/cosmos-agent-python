import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 配置向量存储目录
VECTOR_DB_PATH = "./vector_store"
DATA_PATH = "/Users/wuzexin/Documents"

def ingest_local_data(vector_db_path=VECTOR_DB_PATH, data_path=DATA_PATH):
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return

    # 1️⃣ 初始化 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'mps'}  # 可改为 'cuda' 或 'mps'
    )

    # 2️⃣ 初始化向量数据库
    vector_db = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )

    # 3️⃣ 加载文档
    loader = DirectoryLoader(data_path, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    # 4️⃣ 文档切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    final_docs = splitter.split_documents(docs)

    # 5️⃣ 入库
    vector_db.add_documents(final_docs)
    print(f"✅ Ingested {len(final_docs)} chunks from {data_path}")

if __name__ == "__main__":
    ingest_local_data()