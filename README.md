# Tensor

**Find the gaps. Follow the evidence.**

Tensor is a student preparation assistant built for the NYLP AI Hackathon. It compares a syllabus, study notes and diagnostic quiz answers to help students decide what to study next within their available time.

It tracks two separate kinds of evidence:

- **Notes coverage:** whether the supplied notes contain an explanation of a syllabus topic.
- **Quiz evidence:** what the student's submitted answers demonstrate.

Notes coverage is never treated as proof of understanding. A short quiz or recheck does not establish mastery or prove learning improvement.

## Features

- Paste syllabus and notes text, or upload text-based PDFs.
- Extract syllabus topics with Gemini, then review, edit and confirm them.
- Enter topics manually when extraction is unavailable.
- Inspect notes coverage with source excerpts, verified quote matches and PDF page references where available.
- Take server-scored live practice quizzes, with fixed Java 17 questions where available and labelled AI-generated questions for other topics.
- Work through topic batches while retaining submitted quiz evidence during the session.
- Build a time-bounded study plan with an interactive HTML/SVG roadmap.
- Mark study tasks completed and answer a distinct follow-up question.
- Open an explicitly labelled synthetic Java OOP sample without making AI requests; choose 10 or 20 sample questions.
- Check the AI connection, view processing progress and stop an active analysis.

## Stack

Python, FastAPI, Uvicorn, Pydantic, HTTPX, PyMuPDF, and vanilla HTML/CSS/JavaScript. Live AI uses the Gemini Interactions API. The frontend has no npm build step or external CDN dependency.

## Project structure

```text
app/main.py
static/index.html
requirements.txt
.env.example
.gitignore
README.md
```

The real `.env` file and local virtual environment are intentionally excluded from Git.

## Run locally on Windows PowerShell

Use Python 3.13, as used during local development.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The copy command is for a **new checkout**. Keep an existing configured `.env` when updating your working project.

Edit `.env` locally:

```env
GEMINI_API_KEY=your_private_key
GEMINI_MODEL=gemini-3.8-flash
```

Never put the real key in frontend code, a commit, or a screenshot.

Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Configuration health is available at `/api/health`; API documentation is at `/docs`. The health endpoint does not prove that live model requests will succeed. Use **Check AI connection** for a small live request.

## Try Tensor

### Synthetic sample

Choose **Open sample demo (no AI)**. Explore the report, inspect evidence, take a 10- or 20-question quiz, and open the roadmap and recheck. Sample documents, page references and results are synthetic and remain labelled as such.

### Your own course

1. Enter the course name and available study time.
2. Add the syllabus and notes as pasted text or text-based PDFs.
3. Extract topics, review their scope and confirm the list, or enter topics manually.
4. Analyse notes coverage and compare cited excerpts with the source.
5. Complete a quiz batch, review the report and open the study roadmap.
6. Study a suggested topic and answer its follow-up question.

## Limits and interpretation

- Text-based PDFs: up to 20 MiB and 200 pages each, subject to the extracted-text limits below.
- Syllabus text: up to 120,000 characters; notes text: up to 600,000 characters.
- Up to 100 confirmed topics per report.
- Live quizzes assess up to 20 topics per batch, one question per topic. The sample offers 10 or 20 fixed questions across six Java OOP topics.
- Scanned, handwritten and password-protected PDFs are not supported; OCR is not included.
- Large notes use retrieved candidate passages rather than an exhaustive model reading. Missing evidence in retrieved subsets is marked **Uncertain**.
- Quote verification confirms that cited text exists; it does not certify the interpretation, completeness or factual accuracy of the source.
- AI-generated questions and answer keys are not independently fact-checked. Disputed answers should be checked against course materials.
- Study priorities use transparent rules, not exam predictions or a validated learning-outcome model.
- Frontend progress is temporary. Refreshing or closing the tab clears it. Backend document and quiz sessions are held in memory and expire or disappear when the server restarts.
- Live features require available API access and quota. Provider timeouts, unavailable models and rate limits can prevent analysis. A failed live request is not replaced with fabricated sample results.
- Uploaded PDF text is extracted on the server; relevant extracted text is sent to Google during live AI analysis. Sample mode sends no document content to an AI service.

## Hosting note

Deploy this as a Python web service, not as a static GitHub Pages site. Use **one Uvicorn worker** because jobs, document sessions and quiz sessions are stored in process memory. Public multi-user deployment needs additional authentication, abuse controls and durable storage.

## Verification

During development, controlled tests checked quote matching, PDF page provenance, quiz scoring, time-budget allocation, question batches, cancellation and request deadlines. Mocked model responses were used for automated checks; these are not measurements of model accuracy or student learning gains. Live behaviour depends on the configured model and project quota.
