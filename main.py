import os
import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Gemini API key is missing")
genai.configure(api_key=API_KEY)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def upload_resume():
    return """
    <html>
        <body>
            <h2>Resume ATS Analyzer</h2>
            <form action="/result" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf,image/*,.txt" required>
                <br><br>
                <input type="submit" value="Analyze Resume">
            </form>
        </body>
    </html>
    """


@app.post("/result", response_class=HTMLResponse)
async def result(file: UploadFile):

    file_bytes = await file.read()
    model = genai.GenerativeModel("gemini-flash-latest")

    resume_text = ""
    if file.content_type.startswith("image/"):
        img = Image.open(io.BytesIO(file_bytes))

        vision_prompt = "Extract all readable text from this resume image."
        response = model.generate_content([vision_prompt, img])
        resume_text = response.text

    elif file.content_type == "text/plain":
        resume_text = file_bytes.decode()

    elif file.content_type == "application/pdf":
        # Gemini can read PDFs directly
        response = model.generate_content([
            "Extract all text from this resume PDF",
            {"mime_type": "application/pdf", "data": file_bytes}
        ])
        resume_text = response.text

    else:
        return HTMLResponse("<h3>Unsupported file type</h3>")

    prompt = f"""
You are an ATS (Applicant Tracking System) evaluator.
Analyze the following resume text:

{resume_text}

Provide:
- ATS score (0-100) based on keyword match, clarity, and relevance
- Key strengths
- Potential weaknesses
- Suggested job roles
- Improvements to make resume stronger
"""

    ats_response = model.generate_content(prompt)

    return f"""
    <html>
        <body>
            <h2>ATS Resume Analysis</h2>
            <pre>{ats_response.text}</pre>
        </body>
    </html>
    """