from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from ai_trans.translate_service import setup_translation_chain
from schemas.translate_schema import TranslateRequest, TranslateResponse
import logging
import time
import os, uuid
from PIL import Image, ImageDraw, ImageFont
from utils.whoami import get_model_by_name
from langchain_core.prompts import load_prompt
import ast
from ai_trans.image_translate import ocr_image, clean_content, process_image_with_steps, enhance_image_quality
import cv2
import numpy as np


logger = logging.getLogger(__name__)
translate_router = APIRouter(prefix="/translate")

@translate_router.post("/OpenAI", tags=["Text-Translation"])
async def translate(request: TranslateRequest):
    start_time = time.time()  # 시작 시간 기록

    try:
        print("❇️ Request : ",request)

        chain = setup_translation_chain()

        response = chain.invoke({
            "source_lang": request.source_lang,
            "text": request.text,
            "target_lang": request.target_lang,
        })

        print("✅ Response : ",response)

        end_time = time.time()  # 종료 시간 기록
        latency = (end_time - start_time) * 1000  # 밀리초 단위로 변환
        print(f"OpenAI 실행 시간: {latency:.2f}ms")

        return TranslateResponse(answer=response)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

@translate_router.post("/Image", tags=["Image-Translation"])
async def translate_image(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    model: str = Form(...),
):
    try:
        # 파일 내용을 바이트로 읽기
        contents = await file.read()
        # 바이트를 numpy 배열로 변환
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # image = enhance_image_quality(image)

        # 1. OCR 결과 추출
        ocr_results = ocr_image(image, source_lang, target_lang)
        print("✅ OCR 텍스트 추출 완료! (1/5)")

        # 2. 프롬프트 및 모델 설정
        prompt = load_prompt("./ai_trans/translate_image.yaml", encoding="utf-8")
        llm, model_name = await get_model_by_name(model)
        chain = prompt | llm
        print("✅ 모델 설정 완료! (2/5)")

        # 3. 모델 실행
        translated_texts = chain.invoke({
            "OCR_RESULTS": ocr_results,
            "source_lang": source_lang,
            "target_lang": target_lang
        })
        print("✅ 추출된 텍스트 번역 완료! (3/5)")

        # 4. 결과 정제
        cleaned_translated_texts = clean_content(translated_texts.content)
        print(f"✅ Cleaned Translated Texts : \n{cleaned_translated_texts}")
        cleaned_translated_texts = ast.literal_eval(cleaned_translated_texts)
        print("✅ 결과 정제 완료! (4/5)")

        # 5. 번역된 텍스트 이미지 생성
        pil_image = process_image_with_steps(image, ocr_results, cleaned_translated_texts)
        print("✅ 번역된 텍스트 이미지 생성 완료! (5/5)")

        save_dir = "./translated_images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"translated_{uuid.uuid4().hex}.png")
        pil_image.save(save_path)

        return {"result_image_path": save_path}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

