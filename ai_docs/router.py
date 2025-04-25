from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import io
import logging
from ai_docs.summarize import summarize_file, summarize_combinedTexts, summarize_combined_json, summarize_combined_json_v2, summarize_combined_json_v3
from ai_docs.read_doc import process_file
import json
import time

logger = logging.getLogger(__name__)
document_router = APIRouter(prefix="/document")


@document_router.post("/summarize", tags=["문서 요약"], description=
"""
## VERSION_1
- 각 파일마다 텍스트를 추출 및및 요약을 따로 진행한 후 종합하여 반환
""")
async def summarize_docs_v1(files: list[UploadFile] = File(...)):
    try:
        # 각 파일에 대해 summarize_file 함수를 호출하여 요약을 수행
        summaries = []
        for i, file in enumerate(files):
            summary = await summarize_file(file)
            logger.info(f"✅ 문서 요약 완료 : ({i+1}/{len(files)})")
            summaries.append(summary)
        
        # 종합된 요약을 하나의 문자열로 결합
        combined_summary = "\n\n".join(summaries)

        logger.info(f"✅ 문서요약 종합 완료!")
        logger.info(f"❇️❇️ 종합합 결과 ❇️❇️\n{combined_summary}")
        
        # StreamingResponse로 반환
        return StreamingResponse(io.StringIO(combined_summary), media_type="text/plain")
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    
@document_router.post(
"/summarize_v2", tags=["문서 요약"], description=
"""
## VERSION_2
- 각 파일마다 텍스트를 추출한 후 통합하여 요약 진행
""")
async def summarize_docs_v2(files: list[UploadFile] = File(...)):
    try:
        # 각 파일에 대해 텍스트를 먼저 추출하여 종한한 후 요약을 수행
        extracted_texts = []
        for i, file in enumerate(files):  # enumerate를 사용하여 인덱스와 파일을 가져옵니다.
            extracted_text = await process_file(file)
            print(f"✅ 텍스트 추출 완료 : ({i + 1}/{len(files)})")
            extracted_texts.append(extracted_text)
        
        # 종합된 요약을 하나의 문자열로 결합
        combined_contents = "\n\n\n\n".join(extracted_texts)
        print(f"❇️❇️❇️❇️❇️❇️ 종합된 모든 텍스트 ❇️❇️❇️❇️❇️❇️\n{combined_contents}")

        # combined_contents의 단어 수 계산
        word_count = len(combined_contents.split())
        print(f"✅ 종합된 텍스트의 단어 수: {word_count}")

        summary = await summarize_combinedTexts(combined_contents)
        
        logger.info(f"✅ 문서요약 완료!")
        
        # StreamingResponse로 반환
        return StreamingResponse(io.StringIO(summary), media_type="text/plain")
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    
    
    
@document_router.post(
"/summarize_v3", tags=["문서 요약"], description=
"""
## VERSION_3
- 프롬프트 수정
- 각 파일마다 텍스트 추출하여 json 안에 각 document 별로 요약한 내용 반영하여 요약 진행
""")
async def summarize_docs_v3(files: list[UploadFile] = File(...)):
    start_time = time.time()  # 시작 시간 기록
    try:
        # 각 파일에 대해 텍스트를 먼저 추출하여 종한한 후 요약을 수행
        extracted_texts = {}
        for i, file in enumerate(files):  # enumerate를 사용하여 인덱스와 파일을 가져옵니다.
            extracted_text = await process_file(file)
            print(f"✅ 텍스트 추출 완료 : ({i + 1}/{len(files)})")
            extracted_texts[f'document{i + 1}'] = extracted_text  # JSON 형태로 변환

        # JSON 형태로 변환된 텍스트 출력
        json_contents = json.dumps(extracted_texts, ensure_ascii=False, indent=4)
        print(f"❇️❇️❇️❇️❇️❇️ 종합된 모든 텍스트 (JSON 형태) ❇️❇️❇️❇️❇️❇️\n{json_contents}")

        # JSON 형태의 내용을 모델에 전달하여 요약
        summary = await summarize_combined_json(json_contents)
        
        logger.info(f"✅ 문서요약 완료!")
        
        end_time = time.time()  # 종료 시간 기록
        latency = (end_time - start_time) * 1000  # 밀리초 단위로 변환
        print(f"문서요약 소요 시간: {latency:.2f}ms")

        # StreamingResponse로 반환
        return StreamingResponse(io.StringIO(summary), media_type="text/plain")
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    
@document_router.post(
"/summarize_v4", tags=["문서 요약"], description=
"""
## VERSION_4
- 사용자 요청사항 반영
""")
async def summarize_docs_v4(files: list[UploadFile] = File(...), user_request: str = Form(...)):
    try:
        # 각 파일에 대해 텍스트를 먼저 추출하여 종한한 후 요약을 수행
        extracted_texts = {}
        for i, file in enumerate(files):  # enumerate를 사용하여 인덱스와 파일을 가져옵니다.
            extracted_text = await process_file(file)
            print(f"✅ 텍스트 추출 완료 : ({i + 1}/{len(files)})")
            extracted_texts[f'document{i + 1}'] = extracted_text  # JSON 형태로 변환

        # JSON 형태로 변환된 텍스트 출력
        json_contents = json.dumps(extracted_texts, ensure_ascii=False, indent=4)
        print(f"❇️❇️❇️❇️❇️❇️ 종합된 모든 텍스트 (JSON 형태) ❇️❇️❇️❇️❇️❇️\n{json_contents}")

        # JSON 형태의 내용을 모델에 전달하여 요약
        summary = await summarize_combined_json_v2(json_contents, user_request)
        
        logger.info(f"✅ 문서요약 완료!")
        
        # StreamingResponse로 반환
        return StreamingResponse(io.StringIO(summary), media_type="text/plain")
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")

@document_router.post(
    "/summarize_latest", tags=["문서 요약"], description=
    """
    ## VERSION_5
    - model, length 파라미터 추가
    - 각 파일마다 텍스트 추출하여 json 안에 각 document 별로 요약한 내용 반영하여 요약 진행
    """
)
async def summarize_docs_v5(
    files: list[UploadFile] = File(...),
    model: str = Form(...),     # 모델 이름
    length: str = Form(...)    # 요약 길이
):
    start_time = time.time()  # 시작 시간 기록           
    print(f"✅ 모델: {model}, 요약 길이: {length}")
    try:
        # 각 파일에 대해 텍스트를 먼저 추출하여 종합한 후 요약을 수행
        extracted_texts = {}
        for i, file in enumerate(files):
            extracted_text = await process_file(file)
            print(f"✅ 텍스트 추출 완료 : ({i + 1}/{len(files)})")
            extracted_texts[f'document{i + 1}'] = extracted_text  # JSON 형태로 변환

        # JSON 형태로 변환된 텍스트 출력
        json_contents = json.dumps(extracted_texts, ensure_ascii=False, indent=4)
        # print(f"❇️❇️❇️❇️❇️❇️ 종합된 모든 텍스트 (JSON 형태) ❇️❇️❇️❇️❇️❇️\n{json_contents}")

        # model, length 파라미터 활용하여 요약 함수 호출 (필요시 summarize_combined_json 함수 수정)
        summary = await summarize_combined_json_v3(json_contents, model=model, length=length)
        
        logger.info(f"✅ 문서요약 완료!")
        
        end_time = time.time()  # 종료 시간 기록
        latency = (end_time - start_time) * 1000  # 밀리초 단위로 변환
        print(f"✅ 문서요약 소요 시간: {latency:.2f}ms")

        # StreamingResponse로 반환
        return StreamingResponse(io.StringIO(summary), media_type="text/plain")
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
