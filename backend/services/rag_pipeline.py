import os
import re
from typing import Any, Optional

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms.base import LLM
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

    def _build_llm(self):
        # Try to use OpenAI if API key is available
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            return ChatOpenAI(model_name=model_name, temperature=0.1)

        # Fall back to local Hugging Face model if a token is available
        try:
            from transformers import pipeline

            hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if hf_token:
                os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

            model_name = os.getenv("HF_LLM_MODEL", "distilgpt2")
            task_name = os.getenv("HF_LLM_TASK", "text-generation")
            pipe = pipeline(
                task_name,
                model=model_name,
                tokenizer=model_name,
                device=-1,
                trust_remote_code=False,
            )
            return self._create_transformers_llm(pipe)
        except ImportError:
            return self._create_mock_llm()
        except Exception as exc:
            import logging

            logging.warning(
                f"Failed to initialize local Hugging Face model: {exc}. Using mock LLM."
            )
            return self._create_mock_llm()

    def _create_transformers_llm(self, pipe):
        from langchain.callbacks.manager import CallbackManagerForLLMRun

        class TransformersPipelineLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "transformers"

            def _call(
                self,
                prompt: str,
                stop: Optional[list[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                params = {**kwargs}
                if "temperature" not in params:
                    params["temperature"] = 0.1
                if "max_new_tokens" not in params:
                    params["max_new_tokens"] = 512

                max_input_length = getattr(pipe.tokenizer, "model_max_length", 1024) or 1024
                tokenized = pipe.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_length,
                )
                truncated_prompt = pipe.tokenizer.decode(
                    tokenized["input_ids"][0],
                    skip_special_tokens=True,
                )

                output = pipe(truncated_prompt, **params)
                if isinstance(output, list):
                    first = output[0]
                    if isinstance(first, dict):
                        text = first.get("generated_text") or first.get("text") or str(first)
                    else:
                        text = str(first)
                elif isinstance(output, dict):
                    text = output.get("generated_text") or output.get("text") or str(output)
                else:
                    text = str(output)

                if stop is not None:
                    from langchain.llms.utils import enforce_stop_tokens

                    text = enforce_stop_tokens(text, stop)
                return text

        return TransformersPipelineLLM()

    def _create_mock_llm(self):
        """Create an enhanced mock LLM that extracts info from context."""
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from typing import Optional, List, Any
        import re

        class EnhancedMockLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "enhanced-mock"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                """Extract and summarize information from the context."""
                try:
                    # Extract the context section from the prompt
                    context_match = re.search(r"Context:(.+?)(?:Question:|$)", prompt, re.DOTALL)
                    question_match = re.search(r"Question:(.+?)(?:Answer:|$)", prompt, re.DOTALL)
                    
                    context = context_match.group(1).strip() if context_match else ""
                    question = question_match.group(1).strip() if question_match else ""
                    
                    if not context:
                        return "I don't have any context available to answer this question. Please ensure the repository has been properly ingested."
                    
                    # Extract key information from context
                    lines = [line.strip() for line in context.split('\n') if line.strip()]
                    
                    # Try to find relevant lines that match the question
                    response_lines = []
                    
                    # Keywords from question to match in context
                    question_words = set(word.lower() for word in re.findall(r'\b\w+\b', question))
                    
                    for line in lines:
                        line_lower = line.lower()
                        # If line contains question keywords or seems relevant
                        if any(word in line_lower for word in question_words if len(word) > 3):
                            response_lines.append(line)
                        # Also include lines that look like definitions or descriptions
                        elif any(keyword in line_lower for keyword in ['is', 'are', 'does', 'can', 'provide', 'use', 'help', 'create']):
                            response_lines.append(line)
                    
                    if response_lines:
                        # Take first few relevant lines
                        summary = ' '.join(response_lines[:5])
                        # Limit length
                        if len(summary) > 500:
                            summary = summary[:497] + "..."
                        return summary
                    else:
                        # If no specific matches, provide general info from context
                        return ' '.join(lines[:3]) if lines else "Unable to generate a response based on the available context."
                        
                except Exception as e:
                    return f"Based on the provided context, I can see relevant information in the source files. Please refer to the retrieved documents for specific details."

        return EnhancedMockLLM()

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
        
        # Use default RetrievalQA prompt - it handles context and query properly
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
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
