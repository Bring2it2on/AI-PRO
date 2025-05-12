import asyncio
import websockets
import json
import os
import ssl

ssl_context = ssl._create_unverified_context()

async def main():
    # uri = "ws://localhost:8000/translate/ws/image-translate"
    uri = "wss://dev.chatbaram.com:9000/AI_trans/ws/image-translate"
    file_path = "./ai_trans/test_image/test1.png"
    file_name = os.path.basename(file_path)
    async with websockets.connect(uri, ssl=ssl_context, max_size=2**25) as websocket:
        # 1. 메타데이터 전송 (JSON)
        meta = {
            "file_name": file_name,
            "target_lang": "ko",
            "model": "gpt-4.1",
            "isBold": True
        }
        await websocket.send(json.dumps(meta))
        # 2. 이미지 바이너리 전송
        with open(file_path, "rb") as f:
            await websocket.send(f.read())
        # 3. 단계별 메시지 수신
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"[{data['step']}] {data.keys()}")
                if data['step'] == "ocr":
                    print("OCR 텍스트:", data['ocr_text'])
                    print("감지된 언어:", data['detected_lang'])
                if data['step'] == "done":
                    print("번역 텍스트:", data['translated_texts'])
                    img_data = data['base64_image'].split(",")[1]
                    with open("result.png", "wb") as img_file:
                        img_file.write(base64.b64decode(img_data))
                    print("최종 이미지를 result.png로 저장했습니다.")
                    break
            except websockets.ConnectionClosed:
                print("서버와 연결이 종료되었습니다.")
                break

if __name__ == "__main__":
    import base64
    asyncio.run(main())
