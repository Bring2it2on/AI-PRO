
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from translate.OpenAI.translate_service import setup_translation_chain
from schemas.translate_schema import TranslateRequest, TranslateResponse
import logging

logger = logging.getLogger(__name__)
translate_router = APIRouter(prefix="/translate")

@translate_router.post("/OpenAI", tags=["Text-Translation"])
async def translate(request: TranslateRequest):

    try:
        print("❇️ Request : ",request)

        chain = setup_translation_chain()

        response = chain.invoke({
            "source_lang": request.source_lang,
            "text": request.text,
            "target_lang": request.target_lang,
        })

        print("✅ Response : ",response)

        return TranslateResponse(answer=response)
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @translate_router.post("/LLAMA", tags=["Translation"])
# async def translate(request: TranslateRequest):

#     try:
#         print("번역 요청 발생!! -> ",request)

#         graph = setup_translation_graph_LangGorani()

#         initial_state = {"messages": [HumanMessage(content=request.text)], "targetLanguage" : request.target_lang, "source_lang" : request.source_lang}
        
#         response = graph.invoke(initial_state)

#         print("Response : ",response)

#         translated_text = response["messages"][-1].content

#         print("번역 결과 : ",translated_text)

#         return TranslateResponse(
#             answer=translated_text
#         )
    
#     except Exception as e:
#         logging.error(f"Translation error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))