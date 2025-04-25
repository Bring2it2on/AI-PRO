from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek


async def get_model_by_name(model_name):
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model_name=model_name, temperature=0), model_name
        # return ChatOpenAI(model_name="gpt-4.1", temperature=0), "gpt-4.1"
    elif model_name == "gpt":
        return ChatOpenAI(model_name="gpt-4.1", temperature=0), "gpt-4.1"
    elif model_name.startswith("claude-"):
        return ChatAnthropic(model_name=model_name, temperature=0), model_name
    elif model_name == "claude":
        return ChatAnthropic(model_name="claude-3-5-haiku-latest", temperature=0), "claude-3-5-haiku-latest"
    elif model_name == "deepseek":
        return ChatDeepSeek(model_name="deepseek-chat", temperature=0), "deepseek-chat"
    elif model_name.startswith("deepseek-"):
        return ChatDeepSeek(model_name=model_name, temperature=0), model_name
    else:
        return ChatOpenAI(model_name="gpt-4.1", temperature=0), "gpt-4.1"