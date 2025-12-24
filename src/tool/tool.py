import os
import json
import shutil
import pandas as pd
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2" # 0=INFO, 1=WARNING, 2=ERROR
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
# LangChain Core
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector Store & Retrievers
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.agents import create_pandas_dataframe_agent

# Agent Executor (評測用)
from langchain.agents import AgentExecutor, create_tool_calling_agent

class HybridRAGToolBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ensemble_retriever = None
        
        if os.path.exists(db_path):
            try:
                self._initialize_hybrid_retriever()
            except Exception as e:
                print(f"⚠️ RAG initialization failure: {e}")
        else:
            print(f"⚠️ Unable to locate the vector database path: {db_path}. The RAG functionality will be unavailable.")

    def _initialize_hybrid_retriever(self):

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document"
        )
        
        vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="demo_rag"
        )
        
        existing_data = vector_store.get()
        existing_texts = existing_data['documents']
        existing_metadatas = existing_data['metadatas']

        if not existing_texts:
            print("❌ Chroma DB is empty! Unable to build the RAG tool.")
            return

        # BM25Retriever.from_documents(documents -> List[Document])
        doc_objects = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(existing_texts, existing_metadatas)
        ]

        bm25_retriever = BM25Retriever.from_documents(documents=doc_objects)
        bm25_retriever.k = 5

        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )

    def get_tool(self):
        if not self.ensemble_retriever:
            return None

        @tool
        def search_knowledge_base(query: str) -> str:
            """
            Use this tool to search the “internal knowledge base,” “technical documentation,” or “regulations.”
            It uses hybrid search, which is especially effective for “proper nouns” or “specific concepts.”
            For the input, please provide the complete query question.
            """
            docs = self.ensemble_retriever.invoke(query)
            
            result_contexts = []
            for i, doc in enumerate(docs):
                result_contexts.append(f"--- Document section {i+1} ---\n{doc.page_content}")
            
            return "\n\n".join(result_contexts)

        return search_knowledge_base
    
class SalesDataToolBuilder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            for col in self.df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except:
                        pass
        else:
            print(f"⚠️ Can't find .csv file: {csv_path}")
        
    def get_tool(self, llm):
        """
        這裡需要傳入 llm，因為內部的 Pandas Agent 需要LLM來寫程式
        """
        if self.df is None:
            return None

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=False,
            allow_dangerous_code_execution=True, #需要設定成True，因為它真的在執行 Python
            handle_parsing_errors=True
        )

        @tool
        def analyze_sales_data(query: str) -> str:
            """
            [Function]: A powerful data analysis tool that can write Python code to perform calculations, statistics, grouping, or plotting.
            [When to use]: Use this when the question involves “data computation,” “category comparison,” “trend analysis,” or “which one sells the best.”
            [Input]: Please pass the user’s original question in full.
            """
            try:
                # 呼叫子代理人 (pandas_agent) 分析、統整資料，最後再交由LLM回答
                response = pandas_agent.invoke({"input": query})
                return response['output']
            except Exception as e:
                return f"An error occurred during the analysis process: {str(e)}"
            
        return analyze_sales_data

# Evaluation class  
class TATQABenchmark:
    def __init__(self, dataset_path, temp_dir="./temp_tatqa_context"):
        self.dataset_path = dataset_path
        self.temp_dir = temp_dir
        self.results = []
    
    def _prepare_environment(self, entry):
        if os.path.exists(self.temp_dir):
            try: shutil.rmtree(self.temp_dir)
            except: pass
        os.makedirs(self.temp_dir, exist_ok=True)
        
        csv_path = os.path.join(self.temp_dir, "data.csv")
        txt_path = os.path.join(self.temp_dir, "context.txt")
        db_path = os.path.join(self.temp_dir, "chroma_db")

        # Table -> CSV
        try:
            raw_data = entry["table"]["table"]
            headers = raw_data[0]
            cleaned_headers = []
            seen = {}
            for col in headers:
                if col in seen:
                    seen[col] += 1
                    cleaned_headers.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    cleaned_headers.append(col)
            
            df = pd.DataFrame(raw_data[1:], columns=cleaned_headers)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"❌ CSV Error: {e}")
            return None, None

        # Text -> VectorDB
        try:
            text_list = [p["text"] for p in entry["paragraphs"]]
            full_text = "\n\n".join(text_list)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            loader = TextLoader(txt_path, encoding="utf-8")
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                task_type="retrieval_document"
            )
            Chroma.from_documents(splits, embeddings, persist_directory=db_path, collection_name="demo_rag")
        except Exception as e:
            print(f"❌ RAG Error: {e}")
            return None, None
            
        return csv_path, db_path

    def run(self, llm, limit=3, output_file="./src/eval/raw_benchmark.json"):
            print(f"\n🚀 Starting TAT-QA Generation (Limit: {limit})...")
            
            if not os.path.exists(self.dataset_path):
                print(f"❌ Dataset not found: {self.dataset_path}")
                return

            with open(self.dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            raw_records = []

            for i, entry in enumerate(dataset[:limit]):
                case_id = entry.get("uid", i)
                print(f"\n--- Processing Case {i+1} (ID: {case_id}) ---")
                
                csv_path, db_path = self._prepare_environment(entry)
                if not csv_path: continue
                
                tools = []
                rag = HybridRAGToolBuilder(db_path).get_tool()
                csv = SalesDataToolBuilder(csv_path).get_tool(llm)
                if rag: tools.append(rag)
                if csv: tools.append(csv)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a financial analyst. Use tools to answer questions based on the provided data."),
                    ("user", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
                
                for q_obj in entry["questions"]:
                    q = q_obj["question"]
                    ans = q_obj["answer"]
                    q_type = q_obj.get("answer_from", "unknown")
                    
                    print(f"Query: {q}")
                    try:
                        res = agent_executor.invoke({"input": q})
                        output_text = res['output']
                        print(f"   => Generated: {output_text[:50]}...")
                        
                        raw_records.append({
                            "case_id": case_id,
                            "question": q,
                            "type": q_type,
                            "agent_response": output_text,
                            "ground_truth": ans
                        })
                        
                    except Exception as e:
                        print(f"   => ⚠️ Error: {e}")
                        raw_records.append({
                            "case_id": case_id,
                            "question": q,
                            "type": q_type,
                            "agent_response": f"ERROR: {str(e)}",
                            "ground_truth": ans
                        })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(raw_records, f, ensure_ascii=False, indent=4)
            
            print(f"\n💾 Raw generation results saved to: {output_file}")
            print(f"Total records: {len(raw_records)}")
