from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas.translate_schema import LangDetectRequest, LangDetectResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Translation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # LangSmith 로깅
# from langchain_teddynote import logging
# logging.langsmith("AI-Translation")

# 라우터
from translate.OpenAI.router import translate_router
from ai_docs.router import document_router
from TTS.router import TTS_router

# 라우터 등록
routers = [
    translate_router,
    document_router,
    TTS_router
]

for router in routers:
    app.include_router(router)

# 로거 설정
from logs import LoggerSingleton
import logging

logger = LoggerSingleton.get_logger(
    logger_name="main", level=logging.INFO
)

from langDetect.detector import lang_detector, lang_detector2

@app.post("/lang_detect", tags=["언어감지"])
async def language_detect(request: LangDetectRequest):
    return await lang_detector2(request.INPUT_TEXT)