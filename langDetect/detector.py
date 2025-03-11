from mediapipe.tasks import python
from mediapipe.tasks.python import text

def lang_detector(INPUT_TEXT: str):
    # Create a LanguageDetector object.
    base_options = python.BaseOptions(model_asset_path="detector.tflite")
    options = text.LanguageDetectorOptions(base_options=base_options)
    detector = text.LanguageDetector.create_from_options(options)

    # Get the language detection result for the input text.
    detection_result = detector.detect(INPUT_TEXT)

    # Process the detection result and print the languages detected and their scores.
    if not detection_result.detections:
        return "unknown"  # detections가 비어있는 경우 "unknown" 반환

    for detection in detection_result.detections:
        print(f'{detection.language_code}: ({detection.probability:.2f})')

    return detection_result.detections[0].language_code