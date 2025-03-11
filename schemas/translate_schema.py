from pydantic import BaseModel, Field

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslateResponse(BaseModel):
    answer: str

class LangDetectRequest(BaseModel):
    INPUT_TEXT: str

class LangDetectResponse(BaseModel):
    INPUT_TEXT: str