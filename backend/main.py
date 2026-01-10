from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import json
from agents import HiringAgentWorkflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Hiring Agent")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent workflow
workflow = HiringAgentWorkflow()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze_candidates(
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...)
):
    """
    Analyze candidate resumes against job description
    
    Args:
        job_description: The job posting text
        resumes: List of resume files (PDF or text)
    
    Returns:
        JSON with job analysis and candidate evaluations
    """
    try:
        # Read resume contents
        resume_data = []
        for resume in resumes:
            content = await resume.read()
            resume_data.append({
                "filename": resume.filename,
                "content": content,
                "content_type": resume.content_type
            })
        
        # Run the agent workflow
        result = workflow.process(job_description, resume_data)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
