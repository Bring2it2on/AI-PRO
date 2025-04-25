from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ai_docs.read_doc import process_file
from fastapi import HTTPException, UploadFile
import os
import logging
from utils.whoami import get_model_by_name

def load_prompt_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

logger = logging.getLogger(__name__)

async def summarize_file(file: UploadFile) -> str:
    try:
        # 파일 내용을 처리하여 텍스트 추출
        text = await process_file(file)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "prompts", "AI_DOCS", "summarize.yaml")

        prompt_text = load_prompt_from_file(prompt_path)

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["DOCUMENT_CONTENT"]
        )

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
        chain = prompt | llm

        response = chain.invoke({"DOCUMENT_CONTENT": text})
        
        # 응답에서 요약 추출
        summary = response.content.strip()

        # 마크다운 코드 블록 제거
        if summary.startswith("```markdown\n"):
            summary = summary[len("```markdown\n"):]
        if summary.endswith("```"):
            summary = summary[:-len("```")]

        print(summary)

        return summary
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    
async def summarize_combinedTexts(combined_contents: str) -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "prompts", "AI_DOCS", "summarize_v2.yaml")

        prompt_text = load_prompt_from_file(prompt_path)

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["DOCUMENT_CONTENT"]
        )

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, max_completion_tokens=4096)
        chain = prompt | llm

        response = chain.invoke({"DOCUMENT_CONTENT": combined_contents})
        
        # 응답에서 요약 추출
        summary = response.content.strip()

        # 마크다운 코드 블록 제거
        if summary.startswith("```markdown\n"):
            summary = summary[len("```markdown\n"):]
        if summary.endswith("```"):
            summary = summary[:-len("```")]

        return summary
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    
async def summarize_combined_json(json_contents: str) -> str:
    try:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "prompts", "AI_DOCS", "summarize_v4.yaml")

        prompt_text = load_prompt_from_file(prompt_path)

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["DOCUMENT_CONTENT"]
        )

        llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, max_completion_tokens=4096)
        chain = prompt | llm

        # JSON 내용을 문자열로 변환하여 모델에 전달
        response = chain.invoke({"DOCUMENT_CONTENT": json_contents})
        
        # 응답에서 요약 추출
        summary = response.content.strip()

        # 마크다운 코드 블록 제거
        if summary.startswith("```markdown\n"):
            summary = summary[len("```markdown\n"):]
        if summary.endswith("```"):
            summary = summary[:-len("```")]

        return summary
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=" 문서 요약 중 오류가 발생했습니다.")
    
async def summarize_combined_json_v2(json_contents: str, user_request: str) -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "prompts", "AI_DOCS", "summarize_v5.yaml")

        prompt_text = load_prompt_from_file(prompt_path)

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["DOCUMENT_CONTENT", "USER_REQUEST"]
        )

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_completion_tokens=4096)
        chain = prompt | llm

        # JSON 내용을 문자열로 변환하여 모델에 전달
        response = chain.invoke({"DOCUMENT_CONTENT": json_contents, "USER_REQUEST": user_request})
        
        # 응답에서 요약 추출
        summary = response.content.strip()

        # 마크다운 코드 블록 제거
        if summary.startswith("```markdown\n"):
            summary = summary[len("```markdown\n"):]
        if summary.endswith("```"):
            summary = summary[:-len("```")]

        return summary
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")

def get_summary_length_guide(length: str) -> str:
    if length == "short":
        return (
            "요약을 간결하게 작성하세요. "
            "각 문서의 핵심만 뽑아내어 전체 요약이 너무 길지 않도록 하세요. "
            "문서의 분량이 적을 경우 1~2문장, 보통 분량은 3~5문장 이내로 작성하세요. "
            "여러 문서가 업로드된 경우에도 각 문서별 핵심만 간단히 정리해 전체 요약이 500자 이내가 되도록 하세요. "
            "불필요한 부연설명이나 반복은 피하고, 반드시 중요한 정보만 포함하세요."
        )
    elif length == "long":
        return (
            "요약을 최대한 자세하게 작성하세요. "
            "문서의 분량이 적을 경우에도 내용을 충분히 해설하고, 보통 분량 이상일 경우 10문장 이상, "
            "또는 1000자 이상으로 상세하게 작성하세요. "
            "여러 문서가 업로드된 경우 각 문서별로 주요 내용, 세부 항목, 맥락까지 모두 포함해 종합적으로 정리하세요. "
            "중요한 세부사항, 예시, 맥락, 배경 설명 등도 포함하여, 문서를 읽지 않은 사람도 전체 흐름과 주요 내용을 파악할 수 있도록 하세요."
        )
    else:
        return (
            "문서의 분량과 중요도에 따라 적절한 길이로 요약을 작성하세요. "
            "핵심 내용은 반드시 포함하되, 불필요한 반복이나 장황한 설명은 피하세요."
        )

async def summarize_combined_json_v3(json_contents: str, model: str, length: str) -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "prompts", "AI_DOCS", "summarize_v6.yaml")

        prompt_text = load_prompt_from_file(prompt_path)

        # length 값에 따라 안내문 생성
        summary_length_guide = get_summary_length_guide(length)

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["DOCUMENT_CONTENT", "SUMMARY_LENGTH"]
        )

        llm, _ = await get_model_by_name(model)
        chain = prompt | llm

        response = chain.invoke({
            "DOCUMENT_CONTENT": json_contents,
            "SUMMARY_LENGTH": summary_length_guide
        })
        
        summary = response.content.strip()

        # 마크다운 코드 블록 제거
        if summary.startswith("```markdown\n"):
            summary = summary[len("```markdown\n"):]
        if summary.endswith("```"):
            summary = summary[:-len("```")]

        return summary

    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")

