from fastapi import APIRouter, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from ai_trans.translate_service import setup_translation_chain
from schemas.translate_schema import TranslateRequest, TranslateResponse
import logging
import time, json
import os, uuid, io, base64
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from utils.whoami import get_model_by_name
from langchain_core.prompts import load_prompt
import ast
from ai_trans.image_translate import ocr_image, clean_content, process_image_with_translation, enhance_image_quality, ocr_image_upstage, get_font_path
from langDetect.detector import lang_detector
import cv2
import numpy as np

logger = logging.getLogger(__name__)
translate_router = APIRouter(prefix="/translate")

@translate_router.post("/text", tags=["Text-Translation"])
async def ai_trans(request: TranslateRequest):
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

@translate_router.post("/Image", tags=["Image-Translation"], description=
"""
이미지 번역 서비스

- 이미지 파일을 업로드하면 이미지 안의 텍스트가 번역되어 반환됩니다.
- 이미지 번역 흐름
1. OCR 결과 추출
2. 프롬프트 및 모델 설정
3. 모델 실행(추출된 텍스트 번역)
4. 결과 정제(번역된 텍스트 정제)
5. 번역된 텍스트 이미지 생성

    5-1. 바운딩 박스 영역 마스크 생성 (원본 텍스트 영역 제거) \n
    5-2. 인페인팅으로 텍스트 영역 채우기 (주변 배경과 어우러지도록 채우기) \n
    5-3. 번역된 텍스트 추가 \n
    5-4. 텍스트 위치 조정 (x 좌표가 유사한 바운딩 박스 그룹화) \n
    5-5. 텍스트 그리기 (폰트 크기 조정, 텍스트 색상 추출)
"""                       
)
async def translate_image(
    file: UploadFile = File(...),
    target_lang: str = Form(...),
    model: str = Form(...),
    isBold: bool = Form(False),
):
    try:
        # 파일 내용을 바이트로 읽기
        contents = await file.read()
        filename = file.filename
        # 바이트를 numpy 배열로 변환
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. OCR 결과 추출
        # ocr_results = ocr_image(image, source_lang, target_lang)
        ocr_results, extracted_texts = ocr_image_upstage(contents, filename)
        detected_lang = lang_detector(extracted_texts)
        print("✅ OCR 텍스트 추출 완료! (1/5)")
        print(extracted_texts)

        # 2. 프롬프트 및 모델 설정
        prompt = load_prompt("./ai_trans/translate_image_upstage_v3.yaml", encoding="utf-8")
        llm, model_name = await get_model_by_name(model)
        chain = prompt | llm
        print("✅ 모델 설정 완료! (2/5)")

        # 3. 모델 실행
        translated_texts = chain.invoke({
            "OCR_RESULTS": ocr_results,
            "target_lang": target_lang
        })
        print("✅ 추출된 텍스트 번역 완료! (3/5)")

        # 4. 결과 정제
        cleaned_translated_texts = clean_content(translated_texts.content)
        print("변환 전 문자열 형태:", repr(cleaned_translated_texts))
        
        # 문자열을 Python 객체로 변환
        cleaned_translated_texts = cleaned_translated_texts.strip()
        # 문자열이 올바른 형식인지 확인하고 필요하면 수정
        if not cleaned_translated_texts.startswith('['):
            cleaned_translated_texts = '[' + cleaned_translated_texts
        if not cleaned_translated_texts.endswith(']'):
            cleaned_translated_texts = cleaned_translated_texts + ']'
        cleaned_translated_texts = ast.literal_eval(cleaned_translated_texts)
        joined_translated_texts = "\n".join([text for _, text in cleaned_translated_texts])

        print(f"✅ Cleaned Translated Texts :")
        for i in cleaned_translated_texts:
            print(i)
        print("✅ 번역된 텍스트 정제 완료! (4/5)")

        # 5. 번역된 텍스트 이미지 생성
        font_path = get_font_path(target_lang, isBold)
        translated_image = process_image_with_translation(image, ocr_results, cleaned_translated_texts, font_path)
        print("✅ 번역된 텍스트 이미지 생성 완료! (5/5)")

        # 파일명에서 확장자 추출
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
        format_map = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".bmp": "BMP",
            ".gif": "GIF",
            ".tiff": "TIFF"
        }
        content_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".tiff": "image/tiff"
        }
        save_format = format_map.get(ext.lower(), "PNG")
        content_type = content_type_map.get(ext.lower(), "image/png")

        save_dir = "./ai_trans/translated_images"
        os.makedirs(save_dir, exist_ok=True)
        # 현재 날짜와 시간(예: 2024-05-30_15-22-45)
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(save_dir, f"translated_{now_str}{ext}")
        translated_image.save(save_path, format=save_format)
        print(f"✅ 번역된 이미지 저장 완료! : {save_path}")

        buffered = io.BytesIO()
        translated_image.save(buffered, format=save_format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"base64_image": f"data:{content_type};base64,{img_base64}", 
                "original_text": extracted_texts, 
                "detected_lang": detected_lang,
                "translated_text": joined_translated_texts
                }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")
    
@translate_router.websocket("/ws/image-translate")
async def websocket_translate_image(websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. 첫 메시지: form-data 시뮬레이션 (JSON + 바이너리)
        # 실제로는 JSON 먼저 받고, 그 다음 바이너리(이미지) 받는 식으로 구현
        meta = await websocket.receive_text()
        meta = json.loads(meta)
        file_name = meta["file_name"]
        target_lang = meta["target_lang"]
        model = meta["model"]
        isBold = meta["isBold"]

        # 2. 이미지 바이너리 수신
        image_bytes = await websocket.receive_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. OCR 결과 추출 및 전송
        ocr_results, extracted_texts = ocr_image_upstage(image_bytes, file_name)
        detected_lang = lang_detector(extracted_texts)
        print(f"✅ Detected Language : {detected_lang}")

        await websocket.send_json({
            "step": "ocr",
            "ocr_text": extracted_texts,
            "detected_lang": detected_lang
        })
        
        # 4. 번역 및 모델 실행
        prompt = load_prompt("./ai_trans/translate_image_upstage_v3.yaml", encoding="utf-8")
        llm, model_name = await get_model_by_name(model)
        chain = prompt | llm
        translated_texts = chain.invoke({
            "OCR_RESULTS": ocr_results,
            "target_lang": target_lang
        })

        # 5. 번역 결과 정제
        cleaned_translated_texts = clean_content(translated_texts.content)
        cleaned_translated_texts = cleaned_translated_texts.strip()
        if not cleaned_translated_texts.startswith('['):
            cleaned_translated_texts = '[' + cleaned_translated_texts
        if not cleaned_translated_texts.endswith(']'):
            cleaned_translated_texts = cleaned_translated_texts + ']'
        cleaned_translated_texts = ast.literal_eval(cleaned_translated_texts)

        # 6. 번역된 텍스트 이미지 생성
        font_path = get_font_path(target_lang, isBold)
        translated_image = process_image_with_translation(image, ocr_results, cleaned_translated_texts, font_path)

        # 7. 최종 이미지 base64로 인코딩 후 전송
        buffered = io.BytesIO()
        translated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        joined_translated_texts = "\n".join([text for _, text in cleaned_translated_texts])
        await websocket.send_json({
            "step": "done",
            "base64_image": f"data:image/png;base64,{img_base64}",
            "translated_texts": joined_translated_texts
        })
        await websocket.close()
    except WebSocketDisconnect:
        print("클라이언트 연결 해제")
    except Exception as e:
        await websocket.send_json({"step": "error", "message": str(e)})
        await websocket.close()

