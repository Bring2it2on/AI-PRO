from fastapi import APIRouter, HTTPException
from google.cloud import texttospeech
from pydantic import BaseModel
import random
import base64

# FastAPI 라우터 생성
TTS_router = APIRouter(prefix="/tts", tags=["TTS"])

class TTSRequest(BaseModel):
    text: str
    language_code: str

@TTS_router.post("/synthesize")
async def synthesize_text(request: TTSRequest):
    try:
        # 요청에서 텍스트 추출
        text = request.text
        if not text:
            raise HTTPException(status_code=400, detail="텍스트가 제공되지 않았습니다.")
        
        voice = list_voices(request.language_code)

        # Google Cloud Text-to-Speech 클라이언트 생성
        client = texttospeech.TextToSpeechClient()

        # 요청 생성
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # 음성 설정
        voice = texttospeech.VoiceSelectionParams(
            language_code=request.language_code,
            name="ko-KR-Chirp3-HD-Charon",
            # name=voice.name,
        )

        # 오디오 설정
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
        )

        # API 호출
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # 오디오 파일 저장
        file_name = "./TTS/output.mp3"
        with open(file_name, "wb") as out:
            out.write(response.audio_content)
            print(f"Audio content written to file '{file_name}'")

        # 오디오 파일을 Base64로 인코딩하여 반환
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')

        return {
            "audio_content": audio_base64  # Base64 인코딩된 오디오 데이터
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")
    
def list_voices(language_code):
    client = texttospeech.TextToSpeechClient()

    # API 호출
    response = client.list_voices(language_code=language_code)

    # 지원되는 음성 목록
    voices = response.voices
    if not voices:
        print("No voices found for the specified language code.")
        return None

    # 랜덤으로 하나의 음성 선택
    selected_voice = random.choice(voices)
    print(f"Selected Voice: {selected_voice.name}")

    return selected_voice
