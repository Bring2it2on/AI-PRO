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

class TTSChunkRequest(BaseModel):
    text: str
    language_code: str

class TTSChunkIndexRequest(BaseModel):
    text: str
    language_code: str
    chunk_index: int
    name: str

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
            # name="ko-KR-Chirp3-HD-Charon",
            name=voice.name,
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
        return HTTPException(status_code=400, detail="지원되는 음성이 없습니다.")

    # 랜덤으로 하나의 음성 선택
    selected_voice = random.choice(voices)
    print(f"Selected Voice: {selected_voice.name}")

    return selected_voice

def split_text_by_bytes(text, initial_byte=500, index_byte=3000):
    chunks = []
    current_chunk = ""
    current_bytes = 0
    byte_limit = initial_byte
    char_iter = iter(text)

    for char in char_iter:
        char_bytes = len(char.encode("utf-8"))
        if current_bytes + char_bytes > byte_limit:
            chunks.append(current_chunk)
            current_chunk = char
            current_bytes = char_bytes
            byte_limit = index_byte  # 두 번째 청크부터는 index_byte 사용
        else:
            current_chunk += char
            current_bytes += char_bytes
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

@TTS_router.post("/synthesize_first_chunk")
async def synthesize_first_chunk(request: TTSChunkRequest):
    try:
        # 텍스트 청크 분할 (처음 500byte, 이후 3000byte)
        chunks = split_text_by_bytes(request.text, initial_byte=500, index_byte=3000)
        first_chunk = chunks[0] if chunks else ""
        
        # 언어코드에 맞는 voice 선택
        voice_obj = list_voices(request.language_code)
        print("선택된 음성 정보 : \n",voice_obj)
        print("문자 수 : ",len(first_chunk))
        voice = texttospeech.VoiceSelectionParams(
            language_code=request.language_code,
            name=voice_obj.name,
        )

        # 첫 번째 청크 음성 변환
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=first_chunk)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
        )
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

        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')

        # 청크 메타데이터 생성
        chunk_meta = []
        start = 0
        for idx, chunk in enumerate(chunks):
            end = start + len(chunk.encode("utf-8")) - 1
            chunk_meta.append({"index": idx, "start": start, "end": end})
            start = end + 1

        return {
            "audio_content": audio_base64,
            "chunk_count": len(chunks),
            "chunks": chunk_meta,
            "voice": {
                "language_code": request.language_code,
                "name": voice_obj.name
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

@TTS_router.post("/synthesize_chunk")
async def synthesize_chunk(request: TTSChunkIndexRequest):
    try:
        # 텍스트 청크 분할 (처음 500byte, 이후 3000byte)
        chunks = split_text_by_bytes(request.text, initial_byte=500, index_byte=3000)
        if request.chunk_index < 0 or request.chunk_index >= len(chunks):
            raise HTTPException(status_code=400, detail="유효하지 않은 청크 인덱스입니다.")
        chunk = chunks[request.chunk_index]

        print("문자 수 : ",len(chunk))

        # 언어코드에 맞는 voice 선택
        voice = texttospeech.VoiceSelectionParams(
            language_code=request.language_code,
            name=request.name,
        )

        # 해당 청크 음성 변환
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=chunk)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')

        return {
            "audio_content": audio_base64,
            "index": request.chunk_index
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")
