import os
import re
from typing import Any, Optional

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from backend.services.embeddings import EmbeddingProvider


class RAGPipeline:
    _singleton: Optional["RAGPipeline"] = None

    def __init__(self) -> None:
        self.embedding_provider = EmbeddingProvider()
        self.llm = self._build_llm()
        self.stores: dict[str, FAISS] = {}
        self.chat_history: dict[str, list[dict[str, str]]] = {}

    @classmethod
    def get_instance(cls) -> "RAGPipeline":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def _build_llm(self) -> ChatOpenAI:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the LLM. Please set the environment variable.")

        model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        return ChatOpenAI(model_name=model_name, temperature=0)

    def _split_code(self, document: Document) -> list[Document]:
        text = document.page_content
        split_pattern = r"(?=^(?:def |class |function |const |let |var |export |async |public |private |protected ))"
        chunks: list[Document] = []
        for chunk in re.split(split_pattern, text, flags=re.MULTILINE):
            if not chunk.strip():
                continue
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            for piece in splitter.split_text(chunk):
                chunks.append(Document(page_content=piece, metadata=document.metadata))
        return chunks

    def _split_text(self, document: Document) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents([document])

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            source = document.metadata.get("source", "")
            if source.endswith((".py", ".js")):
                chunks.extend(self._split_code(document))
            else:
                chunks.extend(self._split_text(document))
        return chunks

    def ingest_repository(self, repo_url: str, documents: list[Document]) -> int:
        if not documents:
            raise ValueError("No documents were provided for ingestion.")

        chunks = self._split_documents(documents)
        if not chunks:
            raise ValueError("No document chunks were created during ingestion.")

        embeddings = self.embedding_provider.get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        self.stores[repo_url] = vector_store
        self.chat_history[repo_url] = []
        return len(chunks)

    def query(self, repo_url: str, query: str) -> dict[str, Any]:
        if repo_url not in self.stores:
            raise ValueError("Repository has not been ingested yet. Please call /ingest-repo first.")

        retriever = self.stores[repo_url].as_retriever(search_kwargs={"k": 5})
        prompt_template = """
You are an assistant that answers questions about a GitHub repository based on retrieved source context.
Use the provided context to answer the question. If the answer is unknown, say you don't know instead of fabricating details.

{context}

Question: {query}
Answer in a concise, developer-friendly way and cite source file names when relevant.
"""
        prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        result = chain({"query": query})
        source_documents = result.get("source_documents", [])
        if source_documents is None:
            source_documents = []

        sources = []
        for doc in source_documents:
            source_name = doc.metadata.get("source")
            if source_name and source_name not in sources:
                sources.append(source_name)

        answer = result.get("result", str(result)) if isinstance(result, dict) else str(result)
        self.chat_history[repo_url].append({"question": query, "answer": answer})
        return {"answer": answer, "sources": sources, "history": self.chat_history[repo_url]}
