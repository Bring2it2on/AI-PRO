from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ai_docs.read_doc import process_file
from fastapi import HTTPException, UploadFile
import os
import logging

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
        prompt_path = os.path.join(current_dir, "..", "prompts", "Document_summarize.yaml")

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
        prompt_path = os.path.join(current_dir, "..", "prompts", "Document_summarize_v2.yaml")

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
        prompt_path = os.path.join(current_dir, "..", "prompts", "Document_summarize_v4.yaml")

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
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    
async def summarize_combined_json_v2(json_contents: str, user_request: str) -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "prompts", "Document_summarize_v5.yaml")

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

