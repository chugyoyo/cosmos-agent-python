import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA

# --- 配置 ---
# 1. 待加载文件的本地路径 (请替换为您的实际文件路径)
LOCAL_FILE_PATH = "./my_local_document.txt"
# 2. 您的 Ollama 模型名称 (例如 llama2, gemma, mistral 等)
OLLAMA_MODEL = "llama2"
# 3. Chroma 数据库的存储路径
CHROMA_PERSIST_DIR = "./chroma_db"

# 确保 Chroma 存储目录存在
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# 假设您的本地文件存在，如果没有，请创建一个简单的文本文件用于测试
# 例如，创建一个名为 my_local_document.txt 的文件，内容如下：
# "LangChain 是一个用于开发由语言模型驱动的应用的框架。它提供了一套标准接口和工具集，可以帮助开发者轻松构建 RAG 应用。"
if not os.path.exists(LOCAL_FILE_PATH):
    print(f"警告：文件 {LOCAL_FILE_PATH} 不存在，正在创建示例文件...")
    with open(LOCAL_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("LangChain 是一个用于开发由语言模型驱动的应用的框架。它提供了一套标准接口和工具集，可以帮助开发者轻松构建 RAG 应用。")
    print("示例文件创建成功。")

def setup_rag_system(file_path: str):
    """
    搭建 RAG 系统的核心流程：加载 -> 分块 -> 嵌入 -> 存储。

    """
    print("--- 步骤 1: 加载文档 ---")
    try:
        # 使用 TextLoader 从本地路径加载文件
        # 如果文件是 PDF, DOCX 等，您可能需要使用不同的 loader (如 PyPDFLoader, UnstructuredFileLoader)
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"成功加载文件：{file_path}，包含 {len(documents)} 个文档对象。")
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None

    print("\n--- 步骤 2: 文档分块 (Chunking) ---")
    # 使用 CharacterTextSplitter 将大文档分割成小块 (chunks)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"文档被分割成 {len(texts)} 个文本块。")

    print("\n--- 步骤 3: 向量嵌入 (Embedding) ---")
    # 使用 Ollama 的嵌入模型 (默认使用 Llama2)
    # 确保您的 Ollama 服务正在运行
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    except Exception as e:
        print(f"Ollama 嵌入模型初始化失败，请确保 Ollama 服务正在运行且模型 {OLLAMA_MODEL} 已拉取。错误: {e}")
        return None

    print("\n--- 步骤 4: 存储到向量数据库 ---")
    # 创建并持久化 Chroma 数据库
    # 向量数据库将处理文本块的嵌入并存储起来
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=CHROMA_PERSIST_DIR  # 将数据库内容保存到本地磁盘
    )
    print(f"成功将 {len(texts)} 个文本块存储到 Chroma 数据库 ({CHROMA_PERSIST_DIR})。")
    # 强制保存
    db.persist()

    # 也可以重新加载已有的数据库
    # db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

    return db

def run_query(vector_db: Chroma, query: str):
    """
    运行 RAG 检索和生成流程。
    """
    if vector_db is None:
        print("RAG 系统未成功初始化，无法执行查询。")
        return

    print("\n--- 步骤 5: 检索增强生成 (RAG) ---")

    # 初始化 Ollama LLM
    try:
        llm = Ollama(model=OLLAMA_MODEL)
    except Exception as e:
        print(f"Ollama LLM 初始化失败，请确保 Ollama 服务正在运行。错误: {e}")
        return

    # 创建一个检索器 (Retriever)，它会从向量数据库中找出与查询最相关的文档块
    retriever = vector_db.as_retriever()

    # 创建 RetrievalQA 链，它将整合检索器和 LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将所有检索到的文档块打包到 LLM 的 prompt 中
        retriever=retriever
    )

    print(f"\n-> 查询: {query}")
    # 运行查询
    result = qa_chain.invoke(query)

    print("\n--- 结果 ---")
    print(result['result'])


if __name__ == "__main__":
    print(f"--- 🚀 正在初始化 RAG 系统，使用模型: {OLLAMA_MODEL} ---")

    # 1. 设置 RAG 系统并获取向量数据库实例
    chroma_db_instance = setup_rag_system(LOCAL_FILE_PATH)

    if chroma_db_instance:
        # 2. 运行查询
        test_query = "LangChain 是什么？它的主要作用是什么？"
        run_query(chroma_db_instance, test_query)

    print("\n--- ✅ RAG 系统执行完毕 ---")