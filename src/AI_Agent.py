import os
import warnings

# LangChain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tool.tool import HybridRAGToolBuilder, SalesDataToolBuilder

warnings.filterwarnings("ignore")

if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️ Please set the GOOGLE_API_KEY in your environment variables first.")
    exit(1)

def main():
    print("🧠 Initializing Gemini 2.5 ...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    tools = []
    
    # Hybrid RAG
    rag_builder = HybridRAGToolBuilder("./chroma_db_eng") 
    rag_tool = rag_builder.get_tool()
    if rag_tool:
        tools.append(rag_tool)

    # Python REPL Sales Tool
    csv_builder = SalesDataToolBuilder("./data_source/sales_data.csv")
    csv_tool = csv_builder.get_tool(llm=llm) 
    if csv_tool:
        tools.append(csv_tool)

    if not tools:
        print("❌ Error: No available tools. Program terminated.")
        return

    # prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a smart AI assistant named 'Mengji-Bot'."
            "You have two powerful tools:\n"
            "1. [search_knowledge_base]: For textual knowledge, technical docs, and policies.\n"
            "2. [analyze_sales_data]: For ANY data analysis, math calculation, or finding insights from the sales CSV.\n\n"
            "Strategy:\n"
            "- If the user asks about 'procedures', 'concepts', or 'textual info', use RAG.\n"
            "- If the user asks 'how many', 'sum', 'highest/lowest', 'trend', or 'compare', use analyze_sales_data.\n"
            "- Always think step-by-step."
        )),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("\n=========================================")
    print("🚀 Agent is Ready! (Mode: Code Interpreter + RAG)")
    print("=========================================\n")
    
    while True:
        try:
            q = input("User (輸入 exit 離開) > ")
            if q.lower() in ["exit", "quit"]:
                print("Bye! 👋")
                break
            
            # 執行 Agent
            res = agent_executor.invoke({"input": q})
            print(f"\n🤖 Agent: {res['output']}\n")
            
        except KeyboardInterrupt:
            print("\nForce Close.")
            break
        except Exception as e:
            print(f"❌ System Error: {e}")

if __name__ == "__main__":
    main()