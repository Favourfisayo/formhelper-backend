import base64
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils.error_response import error_response
from utils.embeddings import get_embedding, load_custom_embeddings
from classifier import classify_embedding
from pdf2image import convert_from_bytes
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from export import export_form
from tempfile import NamedTemporaryFile
app = FastAPI()
origins = [
    "https://formhelper.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],           
)
custom_embeddings, custom_labels = load_custom_embeddings()

@app.post("/classify")
async def classify_form(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        upload_emb = get_embedding(file_bytes, file.filename)

        if upload_emb is None:
            return error_response(
                error_type="EMBEDDING_ERROR",
                message="Could not process file into embedding",
                details=f"File: {file.filename}",
                status_code=400,
            )

        result = classify_embedding(upload_emb, custom_embeddings, custom_labels)

        return {
            "success": True,
            "data": result,
            "error": None,
        }

    except Exception as e:
        return error_response(
            error_type="CLASSIFICATION_ERROR",
            message="Unexpected error during classification",
            details=str(e),
            status_code=500,
        )

@app.post("/convert-pdf")
async def convert_pdf_to_image(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "File must be a PDF"})
    try:
        pdf_bytes = await file.read()
        
        image = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)[0]
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {"success": True, "image": img_b64}
    
    except Exception as e:
        return error_response(
            error_type="PDF_CONVERSION_ERROR",
            message="Could not convert pdf file",
            details=f"File: {file.filename}",
            status_code=500,
        )
@app.post("/export-form")
async def export_form_to_pdf(payload: dict):
    try:
        image_url = payload.get("fileUrl")
        fields = payload.get("fields", [])
        print(payload)
        if not image_url or not fields:
            print("Both image_url and fields are required")
            return JSONResponse(
                status_code=400,
                content={"error": "Both image_url and fields are required"},
            )

        resp = requests.get(image_url)
        print(resp)
        if resp.status_code != 200:
            print(f"Failed to download image from {image_url}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to download image from {image_url}"},
            )

        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf_path = export_form(resp.content, fields, tmpfile.name)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="exported_form.pdf",
        )

    except Exception as e:
        return error_response(
            error_type="EXPORT_ERROR",
            message="Could not export filled form",
            details=str(e),
            status_code=500,
        )