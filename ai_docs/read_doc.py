from fastapi import HTTPException, UploadFile
import PyPDF2, docx
import io

def read_pdf(file_bytes):
    file_stream = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file_bytes):
    file_stream = io.BytesIO(file_bytes)
    doc = docx.Document(file_stream)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def process_file(file: UploadFile) -> str:
    """파일 확장자에 따라 파일을 읽고 텍스트를 추출합니다."""
    contents = file.file.read()
    if file.filename.endswith('.pdf'):
        return read_pdf(contents)
    elif file.filename.endswith('.docx'):
        return read_docx(contents)
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")