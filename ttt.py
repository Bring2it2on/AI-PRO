from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import tool
from langchain_openai import ChatOpenAI
import uuid
from langchain_core.runnables import RunnableConfig 
import asyncio

@tool
def law_document(text: str) -> str:
    """법률 문서를 요약합니다."""
    return text

@tool
def report_document(text: str) -> str:
    """보고서 문서를 요약합니다."""
    return text

Tool = [law_document, report_document]
prompt = """문서 형식에 맞게 tool 을 사용하세요"""

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

agent = create_react_agent(
    tools=Tool,
    prompt=prompt,
    llm=llm,
    verbose=True
)

inputs = {

}

config = RunnableConfig(
    recursive_limit = 10,
    configurable = {
        "thread_id" : uuid.uuid4(),
    }
)

async def main():
    prev_node = " "
    async for chunk_msg, metadata in agent.astream(
        inputs,
        config=config,
        stream_mode="messages"
    ):
        curr_node = metadata['langgraph_node']
        if curr_node == 'agent':
            print(chunk_msg)
        else:
            print(chunk_msg)

if __name__ == "__main__":
    asyncio.run(main())


