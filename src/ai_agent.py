import os
import warnings
import logging

# LangChain
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tool.tool import HybridRAGToolBuilder


if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️ Please set the GOOGLE_API_KEY in your environment variables first.")
    exit(1)

def main():
    print("🧠 Initializing Gemini 2.5 ...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    print("💬 Chat Mode...")
    
    tools = []
    
    # Hybrid RAG
    rag_builder = HybridRAGToolBuilder("./chroma_db_eng") 
    rag_tool = rag_builder.get_tool()
    if rag_tool:
        tools.append(rag_tool)

    if not tools:
        print("❌ Error: No available tools. Program terminated.")
        return

    # prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant proficient in analyzing data and retrieving internal knowledge.\n"
            "Currently, You only have access to one powerful tool:\n"
            "1. [search_knowledge_base]: For textual knowledge, technical docs, and policies.\n"
            "Strategy:\n"
            "- If the user asks about 'procedures', 'concepts', or 'textual info', use RAG.\n"
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