import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAGService:
    def __init__(self):
        self.vector_db_path = "../vector_store"
        self.base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        self.lora_path = "../model/lora_adapter"
        self.llm = self._load_finetuned_llm()
        self.retriever = self._load_retriever()

    def _load_finetuned_llm(self):
        print("正在加载微调模型...")
        # 先将模型下载到本地（会自动处理镜像）
        model_dir = snapshot_download(self.base_model_path)

        # 然后从本地路径加载
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype="auto"  # 顺便修复你 log 里的那个 deprecated 警告
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        # 2. 加载 LoRA 适配器 (合并权重)
        # 注意：推理时通常建议 merge_and_unload 以提升速度，或者直接加载
        model = PeftModel.from_pretrained(base_model, self.lora_path)
        model = model.merge_and_unload()  # 合并权重，变成一个普通模型对象

        # 3. 创建 HuggingFace Pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.1
        )

        # 4. 封装为 LangChain LLM
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    def _load_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        vectordb = Chroma(persist_directory=self.vector_db_path, embedding_function=embeddings)
        return vectordb.as_retriever(search_kwargs={"k": 3})

    def get_qa_chain(self):
        # 构建 RAG 链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        return qa_chain


# 初始化单例 (实际生产中建议延迟加载)
rag_service = RAGService()
qa_chain = rag_service.get_qa_chain()