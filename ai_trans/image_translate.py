import easyocr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import requests
import base64
import io
import os
import json
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

def enhance_image_quality(image_input, visualize=True, scale=2, api_token=os.getenv("REPLICATE_API_TOKEN")):
    """
    Replicate Real-ESRGAN API로 업스케일
    Args:
        image_input: str(이미지 경로) 또는 PIL.Image.Image
        visualize: bool - 시각화 여부
        scale: int - 업스케일 배율 (2, 4)
        api_token: str - Replicate API 토큰
    Returns:
        업스케일된 PIL 이미지
    """
    # 이미지 로드 및 base64 인코딩
    if isinstance(image_input, str):
        pil_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        # OpenCV(numpy) 이미지를 PIL 이미지로 변환
        pil_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image_input  # 이미 PIL.Image.Image 타입인 경우
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Replicate API 호출
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    data = {
        "version": "9280e32c0b6e0e6c7e1e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2",  # Real-ESRGAN 최신 버전
        "input": {
            "image": f"data:image/png;base64,{img_str}",
            "scale": scale
        }
    }
    response = requests.post(url, headers=headers, json=data)
    print("응답 내용:", response.text)  # 먼저 출력해서 확인

    if response.status_code == 200 and response.text.strip():
        try:
            prediction = response.json()
        except Exception as e:
            print("JSON 파싱 에러:", e)
            print("응답 내용:", response.text)
    else:
        print("API 응답이 비어있거나 실패했습니다.")

    # 결과 polling
    get_url = prediction["urls"]["get"]
    while True:
        result = requests.get(get_url, headers=headers).json()
        if result["status"] == "succeeded":
            break
        elif result["status"] == "failed":
            raise Exception("업스케일링 실패")
    # 결과 이미지 다운로드
    output_url = result["output"][0]
    out_img = Image.open(requests.get(output_url, stream=True).raw)

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(pil_image)
        plt.title('원본 이미지')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(out_img)
        plt.title('업스케일 결과')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return out_img

def ocr_image(image, source_lang, target_lang):
    reader = easyocr.Reader([source_lang, target_lang])
    ocr_result = reader.readtext(image)
    # 바운딩 박스 좌표를 정수로 변환
    int_ocr_result = []
    for bbox, text, _ in ocr_result:
        int_bbox = [[int(round(x)), int(round(y))] for x, y in bbox]
        int_ocr_result.append((int_bbox, text))

    print("✅ OCR Results :")

    for i in int_ocr_result:
        print(i)
    return int_ocr_result

# 결과 정제
def clean_content(content):
    # 첫 번째 '[' 위치 찾기
    start_idx = content.find('[')
    if start_idx == -1:  # '[' 가 없는 경우
        return content
    
    # '[' 이후의 내용만 추출
    cleaned_content = content[start_idx:]
    return cleaned_content

# 바운딩 박스 영역의 마스크 생성
def create_mask(image, bbox, margin=7):
    """바운딩 박스 영역의 마스크 생성 (마진 추가)"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # bbox 형식 확인
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in bbox)):
        print("잘못된 bbox 형식:", bbox)
        return mask  # 빈 마스크 반환
    
    # 바운딩 박스 좌표를 정수로 변환 및 마진 추가
    int_bbox = [[int(round(x)), int(round(y))] for x, y in bbox]
    expanded_bbox = [
        [max(0, int_bbox[0][0] - margin), max(0, int_bbox[0][1] - margin)],
        [min(image.shape[1], int_bbox[1][0] + margin), max(0, int_bbox[1][1] - margin)],
        [min(image.shape[1], int_bbox[2][0] + margin), min(image.shape[0], int_bbox[2][1] + margin)],
        [max(0, int_bbox[3][0] - margin), min(image.shape[0], int_bbox[3][1] + margin)]
    ]
    
    # 마스크 생성
    pts = np.array(expanded_bbox, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def process_image_with_steps(image, original_results, translated_results):
    # 1. 원본 이미지 로드
    original_image = image.copy()

    # 2. 바운딩 박스 영역 마스크 생성
    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    for bbox in original_results:
        mask = create_mask(original_image, bbox)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 3. 텍스트 영역 제거된 이미지 생성
    masked_image = image.copy()
    masked_image[combined_mask == 255] = [255, 255, 255]  # 흰색으로 채우기
    
    # 4. 인페인팅으로 텍스트 영역 채우기
    inpainted_image = cv2.inpaint(image, combined_mask, 3, cv2.INPAINT_TELEA)

    # 5. 배경 색상 추출
    inpainted_pixels = inpainted_image.reshape(-1, 3)
    background_color = Counter(map(tuple, inpainted_pixels)).most_common(1)[0][0]
    
    # 6. 번역된 텍스트 추가
    translated_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    translated_image = translated_image = add_translated_text(inpainted_image, translated_results, original_image, background_color, fontpath)

    return translated_image

def add_translated_text(inpainted_image, translated_results, original_image, background_color, font_path):
    """번역된 텍스트를 이미지에 추가"""
    translated_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(translated_image)
    
    # 1단계: 바운딩 박스 그룹화
    x_grouped_results = group_bounding_boxes(translated_results)
    
    # 2단계: 텍스트 스타일 추출 및 적용
    extract_text_style_and_draw(draw, original_image, x_grouped_results, background_color, font_path)
    
    return translated_image

def group_bounding_boxes(translated_results, threshold=5):
    """x 좌표가 threshold 이내로 차이나는 바운딩 박스 그룹화"""
    translated_results.sort(key=lambda x: x[0][0][0])  # x 좌표 기준 정렬
    x_grouped_results = []
    current_group = []
    min_x = None
    
    for i, (bbox, translated_text) in enumerate(translated_results):
        current_x = bbox[0][0]
        
        if i == 0 or (current_x - min_x <= threshold):
            current_group.append((bbox, translated_text))
            min_x = min(min_x, current_x) if min_x is not None else current_x
        else:
            x_grouped_results.append((min_x, current_group))
            current_group = [(bbox, translated_text)]
            min_x = current_x
    
    if current_group:
        x_grouped_results.append((min_x, current_group))
    
    return x_grouped_results

def extract_text_style_and_draw(draw, original_image, x_grouped_results, background_color, font_path):
    """원본 텍스트 스타일 추출 및 번역된 텍스트 추가"""
    for min_x, group in x_grouped_results:
        for bbox, translated_text in group:
            # 바운딩 박스 높이와 너비 계산
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # 폰트 크기 조정 (높이 기준)
            font_size = bbox_height - 2  # 약간의 여유를 두기 위해 2를 뺌
            font = ImageFont.truetype(font_path, font_size)

            # 텍스트 크기 측정 및 너비에 맞춰 폰트 조정
            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            while text_width > bbox_width and font_size > 1:
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = draw.textbbox((0, 0), translated_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]

            # 텍스트 색상 추출 (BGR → RGB 변환)
            bgr_color = get_text_color_from_bbox(original_image, bbox)
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])  # BGR → RGB

            print("적용된 텍스트 스타일 : ")
            print("bbox 영역 : ", bbox)
            print("색상 (RGB) : ", rgb_color)
            print("사이즈 : ", font_size)
            print()

            # x 좌표를 min_x로 통일
            adjusted_bbox = [
                [min_x, bbox[0][1]],
                [min_x, bbox[1][1]],
                bbox[2],
                bbox[3]
            ]

            # 텍스트 왼쪽 정렬
            x = adjusted_bbox[0][0]
            y = adjusted_bbox[0][1]
            draw.text((x, y), translated_text, font=font, fill=rgb_color)

def extract_text_style(image, bbox, background_color):
    """원본 텍스트의 스타일(색상, 폰트 크기 등)을 추출하는 함수"""
    # 바운딩 박스 영역 추출
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    text_region = image[y1:y2, x1:x2]

    # text_region 시각화
    plt.imshow(cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB))
    plt.title("Text Region")
    plt.axis('off')
    plt.show()
    
    # RGB에서 HSV로 변환
    text_region_hsv = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)
    background_color_hsv = cv2.cvtColor(np.uint8([[background_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # 색상 추출
    pixels = text_region_hsv.reshape(-1, 3)
    
    # 배경 색상과 유사한 색상 제외
    tolerance = 30  # 허용 오차
    hue_diff = np.abs(pixels[:, 0] - background_color_hsv[0])
    saturation_diff = np.abs(pixels[:, 1] - background_color_hsv[1])
    value_diff = np.abs(pixels[:, 2] - background_color_hsv[2])
    
    mask = (hue_diff > tolerance) | (saturation_diff > tolerance) | (value_diff > tolerance)
    filtered_pixels = pixels[mask]
    print("filtered_pixels : ", filtered_pixels)

    if len(filtered_pixels) == 0:
        return (0, 0, 0), 0  # 색상 추출 실패 시 기본값 반환

    # 가장 많이 나타나는 색상 선택
    most_common_color_hsv = Counter(map(tuple, filtered_pixels)).most_common(1)[0][0]
    most_common_color = cv2.cvtColor(np.uint8([[most_common_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    
    # 폰트 크기 추정 (바운딩 박스 높이 사용)
    font_size = y2 - y1 - 5
    
    return (0,0,0), font_size