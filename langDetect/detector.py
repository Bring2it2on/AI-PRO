from mediapipe.tasks import python
from mediapipe.tasks.python import text
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import time  # 추가

def lang_detector(INPUT_TEXT: str):
    start_time = time.time()  # 시작 시간 기록
    
    # Create a LanguageDetector object.
    model_path = os.path.abspath("detector.tflite")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = text.LanguageDetectorOptions(base_options=base_options)
    detector = text.LanguageDetector.create_from_options(options)

    # Get the language detection result for the input text.
    detection_result = detector.detect(INPUT_TEXT)

    # Process the detection result and print the languages detected and their scores.
    if not detection_result.detections:
        return "unknown"  # detections가 비어있는 경우 "unknown" 반환

    for detection in detection_result.detections:
        print(f'{detection.language_code}: ({detection.probability:.2f})')

    end_time = time.time()  # 종료 시간 기록
    latency = (end_time - start_time) * 1000  # 밀리초 단위로 변환
    print(f"MediaPipe 실행 시간: {latency:.2f}ms")

    return detection_result.detections[0].language_code

def load_prompt_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

async def lang_detector2(INPUT_TEXT: str):
    start_time = time.time()  # 시작 시간 기록
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "..", "prompts", "language_detect_v2.yaml")
    
    prompt_text = load_prompt_from_file(prompt_path)
    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["INPUT_TEXT"]
    )
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = prompt | llm

    result = chain.invoke({"INPUT_TEXT": INPUT_TEXT})
    
    # tl을 fil로 변환
    if result.content.strip() == "tl":
        result.content = "fil"

    end_time = time.time()  # 종료 시간 기록
    latency = (end_time - start_time) * 1000  # 밀리초 단위로 변환
    
    print(f"언어 감지 결과: {result}")
    print(f"실행 시간: {latency:.2f}ms")
    
    return result.content