from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from app.services import process_url, process_pdf, chat_with_content
from app.models import URLRequest, ChatRequest

app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/process_url")
async def api_process_url(url_request: URLRequest):
    try:
        chat_id = process_url(url_request.url)
        return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process_pdf")
async def api_process_pdf(file: UploadFile = File(...)):
    try:
        chat_id = process_pdf(file)
        return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def api_chat(chat_request: ChatRequest):
    response = chat_with_content(chat_request.chat_id, chat_request.question)
    
    if response.startswith("Error:"):
        raise HTTPException(status_code=400, detail=response)
    elif response.startswith("An unexpected error occurred:"):
        raise HTTPException(status_code=500, detail=response)
    else:
        return {"response": f"The main idea of the document is {response}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)