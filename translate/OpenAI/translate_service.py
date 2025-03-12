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
            Act as a professional translator.

            Context(상황):  
            - You are translating text from {source_lang} to {target_lang}.  
            - Ensure the translation is accurate and contextually appropriate.  

            Input Values(입력값):  
            - source_lang: The language of the original text.  
            - target_lang: The language to translate into.  
            - text: The text that needs to be translated.   

            Instructions(단계별 지시사항):  
            - Translate the given text from {source_lang} to {target_lang}.  
            - Maintain the original meaning while ensuring fluency in the target language.  
            - Do not include explanations or additional comments—return only the translated text.  

            Constraints(제약사항):  
            - Output only the translated text in {target_lang}.  
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
                "source_lang": lambda x: x.get("source_lang"),  # 기본값 "ko"
                "target_lang": lambda x: x.get("target_lang"),  # 기본값 "en"
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain
