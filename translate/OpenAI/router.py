from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from translate.OpenAI.translate_service import setup_translation_chain
from schemas.translate_schema import TranslateRequest, TranslateResponse
import logging

logger = logging.getLogger(__name__)
translate_router = APIRouter(prefix="/translate")

@translate_router.post("/OpenAI", tags=["Text-Translation"])
async def translate(request: TranslateRequest):

    try:
        print("❇️ Request : ",request)

        chain = setup_translation_chain()

        response = chain.invoke({
            "source_lang": request.source_lang,
            "text": request.text,
            "target_lang": request.target_lang,
        })

        print("✅ Response : ",response)

        return TranslateResponse(answer=response)
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import easyocr
from fastapi import UploadFile, File
import numpy as np
import cv2
from typing import List, Dict

@translate_router.post("/Image", tags=["Image-Translation"])
async def translate(file: UploadFile = File(...)):
    # 파일 내용을 바이트로 읽기
    contents = await file.read()
    
    # 바이트를 numpy 배열로 변환
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # EasyOCR로 텍스트 인식
    reader = easyocr.Reader(['ko', 'en'])
    ocr_result = reader.readtext(image, paragraph=True)

    print("✅ OCR Result : ",ocr_result)

    # # OCR 결과를 JSON 직렬화 가능한 형식으로 변환
    # result = []
    # for detection in ocr_result:
    #     bbox, text, confidence = detection
    #     result.append({
    #         "bbox": [[float(x) for x in point] for point in bbox],
    #         "text": text,
    #         "confidence": float(confidence) 
    #     })
    
    return
