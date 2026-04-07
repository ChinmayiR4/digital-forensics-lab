---
title: Digital Forensics Lab
emoji: 🕵️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🕵️ Digital Forensics Lab

> **OpenEnv Hackathon Submission** — Multimodal AI Threat Intelligence Environment

An AI agent acts as an **Enterprise Threat Intelligence Analyst** investigating suspicious cases across code, audio, images, video, and hybrid content. The agent must choose forensic tools wisely within a **real budget constraint**, build a case file of evidence, and deliver a final verdict: **is this content AI-generated?**

---

## Why This Environment?

Coordinated disinformation, deepfake fraud, and AI-generated malware are threats that human Trust & Safety teams deal with every day. Existing RL benchmarks test agents on games or toy tasks. This environment tests agents on the **actual multi-step reasoning workflow** an analyst uses:

1. Receive a suspicious case file
2. Decide which forensic tools to use — **budget is limited, choose carefully**
3. Gather and synthesise conflicting evidence (some signals are deliberate traps)
4. Deliver a final verdict with evidence to back it up

A false accusation against a real human costs more than a missed detection. The reward structure reflects this real-world asymmetry — a false positive scores **0.01** — the lowest possible score (strictly above 0 per validator requirement).

---

## Environment Overview

| Property | Value |
|---|---|
| Tasks | 5 (easy → medium → hard → expert → human) |
| Content types | code, audio, image, text, hybrid |
| Action space | 7 forensic tools + 1 verdict action |
| Observation space | CaseFileState (8 fields) |
| Step reward range | −1.0 to +1.0 |
| Task score range | 0.01 to 0.99 (strictly exclusive of 0 and 1) |
| Budget per episode | 1.0 (each tool costs 0.15 — max 6 tool calls) |
| Deterministic | Yes — zero live API calls |

---

## Reward Design

### Layer 1 — Step Rewards (−1.0 to +1.0)

Dense feedback on every action. Teaches the agent good investigative behaviour.

| Action | Reward | Reason |
|---|---|---:|
| Used a **critical** tool (key evidence for this case) | **+0.50** | Key finding |
| Used a **relevant** tool (valid for content type) | **+0.20** | Good progress |
| Used an **irrelevant** tool (wrong for content type) | **−0.20** | Wasted budget, no evidence stored |
| Used an **unknown** action | **−0.20** | Budget still deducted — no free looping |
| Used a tool it already ran (duplicate) | **−0.40** | Redundant, penalised |
| Budget depleted before submitting verdict | **−1.00** | Episode failure |

> **Important:** Irrelevant tools do NOT add to `gathered_evidence`. The step penalty and the grader are fully consistent — a wrong tool hurts in both layers.

### Layer 2 — Task Score (0.1 to 0.99)

The primary evaluation metric. Returned by the grader on `submit_verdict`. Reported in `info["task_score"]`.

| Outcome | Score |
|---|---|
| Correct verdict + all critical evidence cited | **0.99** |
| Correct verdict + ≥50% critical evidence cited | **0.75–0.99** (scaled by ratio) |
| Correct verdict + <50% critical evidence cited | **0.50–0.74** (scaled by ratio) |
| Correct verdict + only non-critical evidence | **0.60** |
| Correct verdict, zero evidence | **0.50** (lucky guess) |
| False negative — missed real AI content | **0.10** |
| False positive — accused human of being AI | **0.01** ← cardinal sin |

### Budget Constraint

Each tool costs **0.15** budget. Budget starts at **1.0**. Running all 7 tools costs **1.05** — more than the available budget. An agent that tries to run every tool will be **forced to end without submitting a verdict** (reward −1.00). The maximum safe number of tool calls before submitting is **6**.

---

## Action Space

### Forensic Tools (cost: 0.15 each)

| Tool | Relevant For | What It Returns |
|---|---|---|
| `analyze_metadata` | all types | EXIF data, file info, creation timestamps |
| `text_perplexity_scan` | code, text, hybrid | Statistical AI-text probability score |
| `check_domain_age` | text, audio, hybrid | Website registration date and age |
| `reverse_image_search` | image, hybrid | Earliest known appearance, GAN fingerprint |
| `run_audio_frequency_scan` | audio, video, hybrid | Synthetic vocal artifact detection |
| `run_ast_code_analyzer` | code, hybrid | Hallucinated library detection, LLM patterns |
| `run_temporal_consistency_check` | video, hybrid | Deepfake physics glitches, lip-sync score |

### Verdict Action

```python
submit_verdict(
    is_ai_generated: bool,          # Your verdict
    confidence: float,              # 0.0 to 1.0
    critical_evidence_keys: list    # List every tool you ran — all of them
)
```

Calling `submit_verdict` ends the episode and triggers the grader. The more critical tools you cite, the higher your `task_score`.

---

## Observation Space

Every call to `reset()` and `step()` returns a `CaseFileState`:

```json
{
  "case_id": "case_001",
  "content_type": "code",
  "content_data": "import hypothetical_lib...",
  "budget_remaining": 0.70,
  "gathered_evidence": {
    "run_ast_code_analyzer": {
      "hallucinated_libraries": ["hypothetical_lib"],
      "ai_probability": 0.92,
      "verdict": "HIGH_CONFIDENCE_AI_GENERATED"
    }
  },
  "step_count": 2,
  "done": false,
  "last_action_error": null
}
```

| Field | Type | Description |
|---|---|---|
| `case_id` | str | Unique case identifier |
| `content_type` | str | code / audio / image / video / text / hybrid |
| `content_data` | str | The suspicious content to investigate |
| `budget_remaining` | float | Starts at 1.0, decreases 0.15 per tool |
| `gathered_evidence` | dict | tool_name → JSON result (relevant tools only) |
| `step_count` | int | Actions taken so far |
| `done` | bool | True when episode ended |
| `last_action_error` | str / null | Error from last action, if any |

---

## Typed Models (OpenEnv Spec)

All three OpenEnv-required typed models are implemented in `env.py`:

```python
# Action model — pass to step()
ForensicAction(action="run_ast_code_analyzer", params={})
ForensicAction(action="submit_verdict", params={
    "is_ai_generated": True,
    "confidence": 0.92,
    "critical_evidence_keys": ["run_ast_code_analyzer"]
})

# Reward model — returned inside ActionResult.reward_detail
ForensicReward(value=0.50, rationale="Critical tool used.")

# Observation model — returned by reset() and step()
CaseFileState(case_id=..., content_type=..., ...)
```

`step()` accepts both a `ForensicAction` model (preferred) and a plain string (backward compatible).

---

## The 5 Tasks

### task_easy — Malicious Code Review
- **Content type:** code
- **Scenario:** A Python script imports `hypothetical_lib` — a library that does not exist on PyPI.
- **The trap:** File metadata looks clean. Only the AST analyzer reveals the smoking gun.
- **Critical tool:** `run_ast_code_analyzer`
- **Baseline score:** 0.99

### task_medium — CEO Voice Clone (BEC Attack)
- **Content type:** audio
- **Scenario:** A voicemail from the "CEO" demands an urgent $50,000 wire transfer.
- **The trap:** Caller ID shows "David Chen — CEO" — looks completely legitimate. The agent must look past it.
- **Critical tool:** `run_audio_frequency_scan` (reveals ElevenLabs v2 signature)
- **Baseline score:** 0.99

### task_hard — Fake Startup Website
- **Content type:** hybrid (text + image + video)
- **Scenario:** A startup's press release, founder headshot, and video pitch.
- **The trap:** `text_perplexity_scan` returns `LIKELY_HUMAN_WRITTEN` — the PR text is genuinely human-authored. An agent that stops here fails.
- **Critical tools:** `reverse_image_search` (StyleGAN face) + `run_temporal_consistency_check` (deepfake video)
- **Baseline score:** 0.99

### task_expert — Fake Social Media Profile
- **Content type:** image
- **Scenario:** Account `@dr_sarah_kim_research`, claims Stanford AI Lab, 3 days old, 0 followers.
- **The trap:** The text sounds credible. The profile picture requires forensic inspection.
- **Critical tools:** `analyze_metadata` (EXIF reveals Stable Diffusion XL) + `reverse_image_search`
- **Baseline score:** 0.99

### task_human — Genuine Human Document
- **Content type:** text
- **Scenario:** An internal legal memo from General Counsel regarding GDPR compliance.
- **The trap:** Everything about this case pattern-matches "AI-suspicious." An agent that blindly flags it as AI-generated scores **0.01** — the lowest possible score (cardinal sin).
- **Critical tool:** `text_perplexity_scan` (high perplexity = human-authored dictation)
- **Baseline score:** 0.99

---

## Baseline Scores (Qwen/Qwen2.5-72B-Instruct)

Scores from a live inference run (Qwen/Qwen2.5-72B-Instruct). All five tasks achieve 0.99 — the maximum possible score under the validator constraint (strictly < 1.0).

| Task | Score | Steps | Key finding |
|---|---|---|---|
| task_easy | 0.99 | 2 | Hallucinated `hypothetical_lib` |
| task_medium | 0.99 | 2 | ElevenLabs v2 vocal signature |
| task_hard | 0.99 | 4 | StyleGAN face + deepfake video |
| task_expert | 0.99 | 3 | EXIF reveals Stable Diffusion XL |
| task_human | 0.99 | 4 | High perplexity = human dictation |

## Project Structure

```
digital-forensics-lab/
├── inference.py        ← Baseline agent (ROOT — mandatory location)
├── main.py             ← FastAPI server (/reset /step /state /health)
├── env.py              ← Core environment logic + typed models + grader
├── mock_database.py    ← Deterministic case data (zero live API calls)
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Production container (port 7860)
├── .dockerignore       ← Keeps venv out of the Docker image
├── openenv.yaml        ← Environment metadata
├── .env.example        ← Template for environment variables
├── .gitignore          ← Protects .env from being committed
├── README.md           ← This file
└── tests/
    └── test_env.py     ← 58 tests covering all tasks and edge cases
```
## Setup
Option 1: Running Locally (Docker)
The environment is fully containerized for reproducible execution.
Bash
docker build -t digital-forensics-lab .
docker run -e HF_TOKEN=your_token_here -p 7860:7860 digital-forensics-lab
Verify the health check at http://localhost:7860/health
Option 2: Running Locally (Python VENV)
Bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
cp .env.example .env      # Add your HF_TOKEN and API configurations
python inference.py       # Run the baseline agent trajectory
Option 3: Hugging Face Spaces Deployment
This environment is designed to be deployed directly as a Hugging Face Docker Space.
1.	Create a new Hugging Face Space (Select Docker SDK).
2.	Connect this GitHub repository to the Space.
3.	In the Space Settings, add your HF_TOKEN as a Repository Secret.
4.	The OpenEnv spec will be accessible via the Space's generated URL.
