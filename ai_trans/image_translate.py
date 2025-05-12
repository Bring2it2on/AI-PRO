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
import mimetypes
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

def ocr_image_upstage(image_bytes, filename):
    """
    이미지 객체를 받아 Upstage OCR API로 텍스트 추출
    """
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY')}"}

    # 파일 확장자에 맞는 mimetype 추출
    mimetype, _ = mimetypes.guess_type(filename)
    if mimetype is None:
        mimetype = "application/octet-stream"  # fallback
    
    # API 요청 - 바이트 스트림 직접 전송
    files = {"document": (filename, image_bytes, mimetype)}
    data = {"model": "ocr"}
    response = requests.post(url, headers=headers, files=files, data=data)
    
    if 'error' in response.json():
        print(f"API 오류: {response.json()['error']['message']}")
        raise ValueError(f"Upstage API 오류: {response.json()['error']['message']}")

    # print("원본 OCR 결과:", response.json())
 
    ocr_results = []
    
    for word in response.json()["pages"][0]["words"]:
        # 바운딩 박스 좌표 추출
        vertices = word["boundingBox"]["vertices"]
        bbox = [[vertex["x"], vertex["y"]] for vertex in vertices]
        text = word["text"]
        
        # 튜플로 묶어서 ocr_results에 추가
        ocr_results.append((bbox, text))
    
    print("원본 OCR 결과:")
    for result in ocr_results:
        print(result)
    
    # 바운딩 박스 통합 처리
    merged_results = merge_adjacent_bboxes(ocr_results)
    
    print("\n통합된 OCR 결과:")
    for result in merged_results:
        print(result)

    # 텍스트 추출
    extracted_texts = "\n".join([text for _, text in merged_results])
    print(extracted_texts)

    return merged_results, extracted_texts

def ocr_image(image, source_lang, target_lang):
    """
    이미지에서 텍스트 추출 및 바운딩 박스 좌표를 정수로 변환
    """
    reader = easyocr.Reader([source_lang, target_lang])
    ocr_result = reader.readtext(image)
    
    # 바운딩 박스 좌표를 정수로 변환
    int_ocr_result = []
    for bbox, text, _ in ocr_result:
        int_bbox = [[int(round(x)), int(round(y))] for x, y in bbox]
        int_ocr_result.append((int_bbox, text))

    print("✅ OCR Results:")
    for i in int_ocr_result:
        print(i)
    return int_ocr_result

def merge_adjacent_bboxes(ocr_results, y_threshold=25, x_threshold=35):
    if not ocr_results:
        return []

    lines = []
    for bbox, text in ocr_results:
        bbox_y_center = sum([p[1] for p in bbox]) / 4
        found = False
        for line in lines:
            all_match = True
            for b, _ in line['items']:
                b_y_center = sum([p[1] for p in b]) / 4
                # y중앙값 차이도 threshold 이내여야 함
                if abs(bbox_y_center - b_y_center) > y_threshold:
                    all_match = False
                    break
                # 둘 중 하나라도 포함되지 않으면 all_match = False
                if not (is_y_center_in_bbox(bbox_y_center, b) or is_y_center_in_bbox(b_y_center, bbox)):
                    all_match = False
                    break
            if all_match:
                line['items'].append((bbox, text))
                found = True
                break
        if not found:
            lines.append({'items': [(bbox, text)]})

    # 이하 x좌표 인접성 병합 로직은 동일
    merged_results = []
    for line in lines:
        print("줄 그룹:", [text for _, text in line['items']])
        items = sorted(line['items'], key=lambda x: x[0][0][0])
        current_group = [items[0]]
        for i in range(1, len(items)):
            prev_bbox = current_group[-1][0]
            curr_bbox = items[i][0]
            if curr_bbox[0][0] - prev_bbox[1][0] <= x_threshold:
                current_group.append(items[i])
            else:
                merged_bbox, merged_text = merge_group(current_group)
                merged_results.append((merged_bbox, merged_text))
                current_group = [items[i]]
        if current_group:
            merged_bbox, merged_text = merge_group(current_group)
            merged_results.append((merged_bbox, merged_text))
    return merged_results

def is_y_center_in_bbox(y_center, bbox):
    y_min = min(p[1] for p in bbox)
    y_max = max(p[1] for p in bbox)
    return y_min <= y_center <= y_max

def merge_group(group):
    """
    그룹 내의 바운딩 박스들과 텍스트를 통합합니다.
    x좌표 순서대로 정렬하여 텍스트를 통합합니다.
    """
    if not group:
        return None, ""
    
    # x좌표 기준으로 그룹 재정렬
    sorted_group = sorted(group, key=lambda x: x[0][0][0])
    
    # 모든 바운딩 박스의 좌표를 고려하여 새로운 바운딩 박스 생성
    all_x = []
    all_y = []
    merged_text = []
    
    for bbox, text in sorted_group:  # sorted_group 사용
        for point in bbox:
            all_x.append(point[0])
            all_y.append(point[1])
        merged_text.append(text)
    
    # 새로운 바운딩 박스 좌표 계산
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    merged_bbox = [
        [min_x, min_y],  # 좌상단
        [max_x, min_y],  # 우상단
        [max_x, max_y],  # 우하단
        [min_x, max_y]   # 좌하단
    ]
    
    # 텍스트 통합 (x좌표 순서대로 정렬된 상태)
    merged_text = " ".join(merged_text)
    
    return merged_bbox, merged_text

def clean_content(content):
    """번역 결과 문자열에서 JSON 형식 추출"""
    start_idx = content.find('[')
    if start_idx == -1:
        return content
    cleaned_content = content[start_idx:]
    return cleaned_content

def create_mask(image, bbox, margin=0):
    """바운딩 박스 영역의 마스크 생성"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # bbox 형식 확인
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in bbox)):
        print("잘못된 bbox 형식:", bbox)
        return mask
    
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

def get_text_color(inpainted_image , image, bbox, contrast_threshold=30):
    """바운딩 박스 영역에서 텍스트 색상 추출"""
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]

    # 인페인팅된 이미지에서 배경색 추출
    inpainted_roi = inpainted_image[y1:y2, x1:x2]
    if inpainted_roi.size == 0:
        return (0, 0, 0)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (0, 0, 0)

    h, w, _ = roi.shape
    center_y = h // 2
    center_row = roi[center_y, :]  # 중앙 row만 추출

    # 색상 카운팅
    reshaped = center_row.reshape(-1, 3)
    color_counts = Counter(map(tuple, reshaped))
    most_common_colors = color_counts.most_common(10)  # 상위 10개

    # 인페인팅된 영역에서 가장 흔한 색상을 배경색으로 사용
    reshaped_inpainted = inpainted_roi.reshape(-1, 3)
    inpainted_color_counts = Counter(map(tuple, reshaped_inpainted))
    bg_color = inpainted_color_counts.most_common(1)[0][0]  # 가장 많이 나타나는 색상

    # 배경색과의 대비 계산
    def contrast(c1, c2):
        return abs(int(np.mean(c1)) - int(np.mean(c2)))

    # 대비 높은 색상 필터링
    filtered_colors = [
        color for color, _ in most_common_colors
        if contrast(color, bg_color) >= contrast_threshold
    ]

    if filtered_colors:
        return filtered_colors[0]  # 대비 높은 첫 번째 색상
    else:
        # 흰색과 검은색 중 배경색과 더 대비되는 색상 선택
        white_contrast = contrast((255, 255, 255), bg_color)
        black_contrast = contrast((0, 0, 0), bg_color)
        return (255, 255, 255) if white_contrast > black_contrast else (0, 0, 0)

def process_image_with_translation(image, original_results, translated_results, font_path):
    """
    이미지 번역 전체 프로세스: 마스킹, 인페인팅, 번역 텍스트 삽입
    """
    # 1. 원본 이미지 복사
    original_image = image.copy()

    # 2. 바운딩 박스 영역 마스크 생성
    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    for bbox, _ in original_results:
        mask = create_mask(original_image, bbox)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 3. 인페인팅으로 텍스트 영역 채우기
    inpainted_image = cv2.inpaint(original_image, combined_mask, 3, cv2.INPAINT_NS)
    
    # 4. 번역된 텍스트 추가
    translated_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(translated_image)
    
    for bbox, translated_text in translated_results:
        # 바운딩 박스 크기 계산
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        bbox_width = abs(x2 - x1)
        bbox_height = abs(y2 - y1)

        # 폰트 크기 조정
        font_size = bbox_height
        font = ImageFont.truetype(font_path, font_size)

        # 텍스트 크기 측정 및 조정
        text_bbox = draw.textbbox((0, 0), translated_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        while text_width > bbox_width + 4 and font_size > 1:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]

        # 텍스트 색상 추출
        bgr_color = get_text_color(inpainted_image, original_image, bbox)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])  # BGR → RGB

        print(f"좌표: {bbox}, 텍스트: '{translated_text}', 색상: {rgb_color}, 크기: {font_size}")

        # 텍스트 위치 조정 (바운딩 박스의 x, y 사용)
        x = x1
        y = y1 + (bbox_height - text_height) // 2  # 중앙정렬
        
        # 텍스트 그리기
        draw.text((x, y), translated_text, font=font, fill=rgb_color)

    return translated_image

def enhance_image_quality(image_input, visualize=True, scale=2, api_token=os.getenv("REPLICATE_API_TOKEN")):
    """
    Replicate API를 통한 이미지 업스케일링
    """
    # 이미지 로드 및 base64 인코딩
    if isinstance(image_input, str):
        pil_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image_input
        
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
        "version": "9280e32c0b6e0e6c7e1e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2e7e2",
        "input": {
            "image": f"data:image/png;base64,{img_str}",
            "scale": scale
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200 or not response.text.strip():
        print("API 응답 실패:", response.text)
        return pil_image  # 원본 이미지 반환
        
    try:
        prediction = response.json()
        get_url = prediction["urls"]["get"]
        
        # 결과 polling
        while True:
            result = requests.get(get_url, headers=headers).json()
            if result["status"] == "succeeded":
                break
            elif result["status"] == "failed":
                return pil_image  # 실패 시 원본 반환
                
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
        
    except Exception as e:
        print("처리 중 오류 발생:", str(e))
        return pil_image  # 오류 시 원본 반환
    
def get_font_path(target_lang, isBold=False):
    """
    대상 언어에 따라 적절한 폰트 경로를 반환합니다.
    
    Args:
        target_lang (str): 대상 언어 코드 (예: 'en', 'ko', 'ja', 'zh')
        
    Returns:
        str: 폰트 파일 경로
    """
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "fonts"
    
    # 언어별 폰트 매핑
    font_map = {
        'en': {
            'regular': "NotoSans-Regular.ttf",
            'bold': "NotoSans-Bold.ttf"
        },
        'ko': {
            'regular': "NotoSansKR-Regular.ttf",
            'bold': "NotoSansKR-Bold.ttf"
        },
        'ja': {
            'regular': "NotoSerifJP-Regular.ttf",
            'bold': "NotoSerifJP-Bold.ttf"
        },
        'zh-Hans': {
            'regular': "NotoSerifSC-Regular.ttf",
            'bold': "NotoSerifSC-Bold.ttf"
        },
        'zh-Hant': {
            'regular': "NotoSerifTC-Regular.ttf",
            'bold': "NotoSerifTC-Bold.ttf"
        }
    }
    
    # 기본 폰트 (없는 언어용)
    default_font = {
        'regular': "NotoSans-Regular.ttf",
        'bold': "NotoSans-Bold.ttf"
    }
    
    font_name = font_map.get(target_lang, default_font)[isBold and 'bold' or 'regular']
    
    font_path = base_dir / font_name
    
    # 폰트 파일이 없으면 기본 폰트 사용
    if not font_path.exists():
        font_path = base_dir / default_font[isBold and 'bold' or 'regular']

    print(f"폰트 경로: {font_path}")
    
    return str(font_path)