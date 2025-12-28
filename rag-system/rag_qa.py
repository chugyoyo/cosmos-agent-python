import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

VECTOR_DB_PATH = "./vector_store"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

class RAGService:
    def __init__(self, vector_db_path=VECTOR_DB_PATH):
        self.vector_db_path = vector_db_path

        # 设备选择
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"🖥️ Using device: {self.device}")

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': self.device}
        )

        # LLM
        self.llm = self._load_model()

        # 向量库
        self.vector_db = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )

    def _load_model(self):
        print("📥 Downloading/Loading Model...")
        model_dir = snapshot_download(BASE_MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)

    def get_chain(self):
        system_prompt = (
            "你是一个专业的助手。请仅根据提供的上下文（Context）回答问题。"
            "如果你在上下文中找不到答案，请诚实告知。回答请简明扼要。"
            "\n\n上下文: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        return create_retrieval_chain(
            self.vector_db.as_retriever(search_kwargs={"k": 3}),
            question_answer_chain
        )

if __name__ == "__main__":
    service = RAGService()
    rag_chain = service.get_chain()

    user_input = "我有哪些比较薄弱的后端开发知识点？"
    response = rag_chain.invoke({"input": user_input})

    print("\n🤖 AI Answer:\n", response["answer"])
    print("\n📄 Sources used:", [d.id for d in response["context"]])
