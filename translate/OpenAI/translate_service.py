from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 로드
load_dotenv()

def setup_translation_chain():
    """
    GPT 번역 체인 설정 함수
    """
    # 번역 프롬프트 1
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", """
    #         # Task:
    #         - Translation

    #         # Instructions:
    #         - Translate the text from {source_lang} to {target_lang}.
    #         - Ensure the translation is contextually accurate.
    #         - Just return the translated result only in {target_lang}.
    #         """),
    #         ("human", "Text: {text}")
    #     ]
    # )

    # 번역 프롬프트 2
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                Role(역할지정):  
                Act as a professional translator with contextual awareness.

                Input Description(입력값 설명):  
                - text: The text that needs to be translated.   
                - retriever: Additional retrieved information to provide context or explain terminology.

                Input Values(입력값):  
                - retriever: {retriever}

                Context(상황):  
                - You are a translation expert specializing in multilingual text conversion.
                - The source and target languages will be provided using ISO 639-1 language codes.
                - Additional context or terminology explanations may be provided via `retriever` to enhance translation accuracy. 
                - Your task is to translate text from `{source_lang}` to `{target_lang}` while ensuring contextual accuracy and fluency.

                Instructions(단계별 지시사항):
                - Identify `{source_lang}` and `{target_lang}` as ISO 639-1 language codes.
                - Translate text from `{source_lang}` to `{target_lang}` while maintaining meaning, fluency, and cultural relevance.
                - If `retriever` is provided, use it to enhance translation accuracy by incorporating relevant context or clarifying terminology.  
                - Return only the translated text in `{target_lang}` without additional comments or explanations.  

                Constraints(제약사항):  
                - Output only the translated text in `{target_lang}`.
                - If `retriever` contains relevant context, apply it thoughtfully to improve the translation.  
                - Do not output `retriever` directly; instead, use it to refine the translation.   
                - Avoid unnecessary modifications unless needed for clarity.  

                Output Indicator(출력값 지정):  
                - Output format: Plain Text  
                - Output fields: Translated text only   
            """),
            ("human", "{text}")
        ]
    )

    # OpenAI LLM 모델 설정 (gpt-4o-mini 사용)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # 체인 설정
    chain = (
            {
                "text": lambda x: x["text"],  # 올바른 lambda 구문
                "source_lang": lambda x: x.get("source_lang"),
                "target_lang": lambda x: x.get("target_lang"), 
                "retriever": lambda x: x.get("retriever", None),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    

    return chain
