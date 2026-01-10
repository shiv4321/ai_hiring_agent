import os
import json
import re
from typing import List, Dict, Any, TypedDict
from groq import Groq
import PyPDF2
from io import BytesIO
from langgraph.graph import StateGraph, END

# Initialize Groq client
# Initialize Groq client with environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=GROQ_API_KEY)

class WorkflowState(TypedDict):
    """State schema for the workflow"""
    job_description: str
    resumes: List[Dict[str, Any]]
    job_analysis: Dict[str, Any]
    candidate_evaluations: List[Dict[str, Any]]

class HiringAgentWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self):
        self.model = "llama-3.3-70b-versatile"
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("parse_resumes", self._parse_resumes)
        workflow.add_node("analyze_job", self._analyze_job)
        workflow.add_node("evaluate_candidates", self._evaluate_candidates)
        workflow.add_node("generate_questions", self._generate_questions)
        
        # Define edges
        workflow.set_entry_point("parse_resumes")
        workflow.add_edge("parse_resumes", "analyze_job")
        workflow.add_edge("analyze_job", "evaluate_candidates")
        workflow.add_edge("evaluate_candidates", "generate_questions")
        workflow.add_edge("generate_questions", END)
        
        return workflow.compile()
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"[Error extracting PDF: {str(e)}]"
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call Groq LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _parse_resumes(self, state: WorkflowState) -> WorkflowState:
        """Agent: Parse and extract structured info from resumes"""
        parsed_resumes = []
        
        for resume in state["resumes"]:
            # Extract text based on content type
            if resume["content_type"] == "application/pdf":
                text = self._extract_text_from_pdf(resume["content"])
            else:
                text = resume["content"].decode("utf-8")
            
            # Use LLM to extract structured information
            system_prompt = """You are a resume parser. Extract key information from resumes and return it in JSON format.
Focus on concrete details, not just keywords. Look for evidence of actual work and accomplishments."""
            
            prompt = f"""Parse this resume and extract the following information in JSON format:
{{
  "name": "candidate name",
  "email": "email address",
  "phone": "phone number",
  "experience_years": number,
  "skills": ["list of skills with context"],
  "experience": ["list of work experiences with concrete details"],
  "education": ["educational background"],
  "projects": ["notable projects with outcomes"],
  "summary": "brief professional summary"
}}

Resume text:
{text[:4000]}

Return ONLY the JSON object, no other text."""
            
            try:
                response = self._call_llm(prompt, system_prompt)
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group())
                    parsed_data["filename"] = resume["filename"]
                    parsed_data["raw_text"] = text[:2000]  # Keep sample for analysis
                    parsed_resumes.append(parsed_data)
            except Exception as e:
                parsed_resumes.append({
                    "filename": resume["filename"],
                    "error": str(e),
                    "raw_text": text[:2000]
                })
        
        state["resumes"] = parsed_resumes
        return state
    
    def _analyze_job(self, state: WorkflowState) -> WorkflowState:
        """Agent: Analyze job description and extract requirements"""
        system_prompt = """You are a job requirements analyzer. Extract key requirements from job descriptions.
Focus on must-have skills, experience levels, and important qualifications."""
        
        prompt = f"""Analyze this job description and extract requirements in JSON format:
{{
  "title": "job title",
  "experience_required": number of years,
  "required_skills": ["list of must-have skills"],
  "preferred_skills": ["nice-to-have skills"],
  "education_requirements": ["education requirements"],
  "key_responsibilities": ["main responsibilities"],
  "evaluation_criteria": ["what matters most for this role"]
}}

Job Description:
{state['job_description'][:3000]}

Return ONLY the JSON object, no other text."""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                job_analysis = json.loads(json_match.group())
                state["job_analysis"] = job_analysis
        except Exception as e:
            state["job_analysis"] = {"error": str(e)}
        
        return state
    
    def _evaluate_candidates(self, state: WorkflowState) -> WorkflowState:
        """Agent: Evaluate each candidate against job requirements"""
        job_analysis = state["job_analysis"]
        evaluations = []
        
        system_prompt = """You are an expert technical recruiter. Evaluate candidates based on:
1. DEPTH over keywords - look for concrete examples and achievements
2. RELEVANCE - how well experience matches the role
3. CONSISTENCY - check for logical career progression
4. RED FLAGS - identify vague claims or keyword stuffing

Score candidates objectively based on evidence, not just keyword matches."""
        
        for resume in state["resumes"]:
            if "error" in resume:
                evaluations.append({
                    "filename": resume["filename"],
                    "error": resume["error"]
                })
                continue
            
            prompt = f"""Evaluate this candidate for the job and return a JSON evaluation:
{{
  "name": "{resume.get('name', 'Unknown')}",
  "email": "{resume.get('email', 'N/A')}",
  "score": total score (0-100),
  "breakdown": {{
    "experience": score 0-30 (relevance and depth of experience),
    "skills": score 0-30 (technical skills match),
    "education": score 0-20 (educational fit),
    "overall_fit": score 0-20 (career trajectory, cultural fit)
  }},
  "strengths": ["list 3-5 key strengths with specific examples"],
  "gaps": ["list 2-4 knowledge gaps or concerns"],
  "red_flags": ["any concerns about keyword stuffing or vague claims"],
  "reasoning": "detailed explanation of scoring with specific evidence"
}}

Job Requirements:
- Title: {job_analysis.get('title', 'N/A')}
- Experience: {job_analysis.get('experience_required', 'N/A')} years
- Required Skills: {', '.join(job_analysis.get('required_skills', []))}
- Key Responsibilities: {', '.join(job_analysis.get('key_responsibilities', [])[:3])}

Candidate Resume:
Name: {resume.get('name', 'Unknown')}
Experience: {resume.get('experience_years', 'N/A')} years
Skills: {', '.join([str(s) if isinstance(s, dict) else s for s in resume.get('skills', [])[:10]])}
Experience Details: {' | '.join([str(e) if isinstance(e, dict) else e for e in resume.get('experience', [])[:3]])}
Education: {', '.join([str(e) if isinstance(e, dict) else e for e in resume.get('education', [])])}IMPORTANT: Look for concrete examples and achievements, not just keyword lists. 
Penalize vague claims without supporting details.

Return ONLY the JSON object, no other text."""
            
            try:
                response = self._call_llm(prompt, system_prompt)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                    evaluation["filename"] = resume["filename"]
                    evaluations.append(evaluation)
            except Exception as e:
                evaluations.append({
                    "filename": resume["filename"],
                    "name": resume.get("name", "Unknown"),
                    "error": str(e)
                })
        
        state["candidate_evaluations"] = evaluations
        return state
    
    def _generate_questions(self, state: WorkflowState) -> WorkflowState:
        """Agent: Generate targeted interview questions for each candidate"""
        job_analysis = state["job_analysis"]
        
        system_prompt = """You are an expert interviewer. Generate probing questions that:
1. Verify depth of claimed skills
2. Explore gaps or concerns
3. Assess problem-solving ability
4. Check cultural and role fit

Questions should be specific to the candidate's background."""
        
        for evaluation in state["candidate_evaluations"]:
            if "error" in evaluation:
                continue
            
            prompt = f"""Generate 5-7 targeted interview questions for this candidate.

Job: {job_analysis.get('title', 'N/A')}
Candidate: {evaluation.get('name', 'Unknown')}
Score: {evaluation.get('score', 'N/A')}/100
Strengths: {', '.join(evaluation.get('strengths', []))}
Gaps: {', '.join(evaluation.get('gaps', []))}

Return as a JSON array of strings:
["Question 1?", "Question 2?", ...]

Focus on:
- Verifying specific skills mentioned in their resume
- Exploring their knowledge gaps
- Understanding their problem-solving approach
- Assessing their fit for key responsibilities

Return ONLY the JSON array, no other text."""
            
            try:
                response = self._call_llm(prompt, system_prompt)
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    questions = json.loads(json_match.group())
                    evaluation["interview_questions"] = questions
                else:
                    evaluation["interview_questions"] = []
            except Exception as e:
                evaluation["interview_questions"] = [f"Error generating questions: {str(e)}"]
        
        return state
    
    def process(self, job_description: str, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process job description and resumes through the workflow"""
        initial_state = {
            "job_description": job_description,
            "resumes": resumes,
            "job_analysis": {},
            "candidate_evaluations": []
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Format output
        result = {
            "job_analysis": final_state["job_analysis"],
            "candidates": sorted(
                final_state["candidate_evaluations"],
                key=lambda x: x.get("score", 0),
                reverse=True
            )
        }
        
        return result
