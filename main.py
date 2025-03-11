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

# LangSmith 로깅
from langchain_teddynote import logging
logging.langsmith("AI-Translation")

# 라우터
from translate.OpenAI.router import translate_router

# 라우터 등록
routers = [
    translate_router
]

for router in routers:
    app.include_router(router)

# 로거 설정
from logs import LoggerSingleton
import logging

logger = LoggerSingleton.get_logger(
    logger_name="main", level=logging.INFO
)

from langDetect.detector import lang_detector

@app.post("/lang_detect", tags=["언어감지"])
async def language_detect(request: LangDetectRequest):
    return lang_detector(request.INPUT_TEXT)