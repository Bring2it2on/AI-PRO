from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import io, os
import logging
from summarize import summarize

logger = logging.getLogger(__name__)
document_router = APIRouter(prefix="/document")


@document_router.post("/summarize", tags=["문서 요약"])
async def summarize_docs(files: list[UploadFile] = File(...)):
    try:
        # 각 파일에 대해 summarize2 함수를 호출하여 요약을 수행
        summaries = []
        for file in files:
            summary = await summarize(file)
            summaries.append(summary)
        
        # 종합된 요약을 하나의 문자열로 결합
        combined_summary = "\n\n".join(summaries)
        
        # StreamingResponse로 반환
        return StreamingResponse(io.StringIO(combined_summary), media_type="text/plain")
    
    except Exception as e:
        logger.info(f"문서 요약 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="문서 요약 중 오류가 발생했습니다.")
    