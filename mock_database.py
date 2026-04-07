"""
mock_database.py - Deterministic Case Data for Digital Forensics Lab

All tool responses are hardcoded. Zero live API calls.
5 cases: easy (code), medium (audio), hard (hybrid), expert (image), human (text)

Design notes:
  - Only RELEVANT tools have responses. Irrelevant tools are blocked at the
    env layer (env.py _use_tool) and will not reach this data.
  - Every content_type=hybrid case includes responses for all 7 tools
    since hybrid content is relevant for all tools.
"""

CASES = {
    "task_easy": {
        "case_id": "case_001",
        "content_type": "code",
        "content_data": (
            "import requests\n"
            "from transformers import AutoTokenizer\n"
            "from langchain_openai import ChatOpenAI\n"
            "from hypothetical_lib import SmartSummarizer\n\n"
            "def analyze_sentiment(text: str) -> dict:\n"
            "    summarizer = SmartSummarizer()\n"
            "    return summarizer.process(text)\n"
        ),
        "ground_truth": True,
        "critical_evidence": ["run_ast_code_analyzer"],
        # Only tools relevant for 'code': analyze_metadata, run_ast_code_analyzer,
        # text_perplexity_scan
        "tool_responses": {
            "analyze_metadata": {
                "file_type": "python",
                "author": "unknown",
                "suspicious": False,
                "note": "Metadata looks clean — not the critical clue",
            },
            "run_ast_code_analyzer": {
                "hallucinated_libraries": ["hypothetical_lib"],
                "suspicious_patterns": ["SmartSummarizer not in PyPI"],
                "ai_probability": 0.92,
                "verdict": "HIGH_CONFIDENCE_AI_GENERATED",
            },
            "text_perplexity_scan": {
                "perplexity_score": 45.2,
                "ai_probability": 0.60,
                "verdict": "POSSIBLY_AI_GENERATED",
            },
        },
    },

    "task_medium": {
        "case_id": "case_002",
        "content_type": "audio",
        "content_data": (
            "Voicemail: 'Hi, this is David Chen CEO. Wire $50,000 urgently "
            "to account 8847291023. Time-sensitive. Do not discuss. "
            "Call +1-555-0199.'"
        ),
        "ground_truth": True,
        "critical_evidence": ["run_audio_frequency_scan"],
        # Relevant for 'audio': analyze_metadata, run_audio_frequency_scan,
        # text_perplexity_scan, check_domain_age
        "tool_responses": {
            "analyze_metadata": {
                "file_type": "audio/mp3",
                "caller_id": "David Chen - CEO",
                "suspicious": False,
                "note": "Caller ID looks legit — THE TRAP",
            },
            "run_audio_frequency_scan": {
                "synthetic_vocal_artifacts": True,
                "mel_spectrogram_anomalies": ["Unnatural pitch at 0.3s, 4.1s"],
                "known_voice_cloning_signature": "ElevenLabs v2 detected",
                "ai_probability": 0.97,
                "verdict": "HIGH_CONFIDENCE_VOICE_CLONE",
            },
            "text_perplexity_scan": {
                "perplexity_score": 38.1,
                "ai_probability": 0.55,
                "verdict": "POSSIBLY_AI_GENERATED",
            },
            "check_domain_age": {
                "verdict": "NOT_APPLICABLE",
                "note": "No external domain associated with this voicemail.",
            },
        },
    },

    "task_hard": {
        "case_id": "case_003",
        "content_type": "hybrid",
        "content_data": (
            "NovaTech AI Solutions press release. CEO Dr. Alexandra Rivers. "
            "$10M Series A. [Video pitch] [Headshot photo]"
        ),
        "ground_truth": True,
        "critical_evidence": ["reverse_image_search", "run_temporal_consistency_check"],
        # Relevant for 'hybrid': all 7 tools — every tool has a response
        "tool_responses": {
            "analyze_metadata": {
                "ssl_cert_age_days": 3,
                "suspicious": True,
                "note": "New domain — weak signal alone, needs corroboration",
            },
            "text_perplexity_scan": {
                "perplexity_score": 72.1,
                "ai_probability": 0.31,
                "verdict": "LIKELY_HUMAN_WRITTEN",
                "note": "THE TRAP — human-written PR text. Do not stop here.",
            },
            "reverse_image_search": {
                "stylegan_fingerprint": True,
                "ai_generated_probability": 0.95,
                "verdict": "STYLEGAN_GENERATED_FACE",
            },
            "run_temporal_consistency_check": {
                "physics_glitches_detected": True,
                "background_warping": ["Frame 47-52: ripple artifact"],
                "lip_sync_score": 0.41,
                "deepfake_probability": 0.91,
                "verdict": "HIGH_CONFIDENCE_DEEPFAKE_VIDEO",
            },
            "check_domain_age": {
                "domain": "novatech-ai.io",
                "age_days": 3,
                "verdict": "NEWLY_REGISTERED_SUSPICIOUS",
            },
            "run_audio_frequency_scan": {
                "synthetic_vocal_artifacts": False,
                "ai_probability": 0.20,
                "verdict": "LIKELY_HUMAN_VOICE",
                "note": "Red herring — audio appears genuine",
            },
            # Added: run_ast_code_analyzer is relevant for hybrid content;
            # agents exploring it should get a meaningful (non-empty) response.
            "run_ast_code_analyzer": {
                "verdict": "NOT_APPLICABLE",
                "note": (
                    "No standalone code files found. Site uses standard "
                    "HTML/CSS/JS — no LLM hallucination patterns in client scripts."
                ),
                "ai_probability": 0.0,
            },
        },
    },

    "task_expert": {
        "case_id": "case_004",
        "content_type": "image",
        "content_data": (
            "Social media profile @dr_sarah_kim_research. Claims Stanford AI Lab. "
            "Account 3 days old, 0 followers, 847 following. "
            "Posts inflammatory AI research content."
        ),
        "ground_truth": True,
        "critical_evidence": ["analyze_metadata", "reverse_image_search"],
        # Relevant for 'image': analyze_metadata, reverse_image_search
        "tool_responses": {
            "analyze_metadata": {
                "exif_software": "Stable Diffusion XL 1.0",
                "camera_make": None,
                "gps_data": None,
                "suspicious": True,
                "note": "EXIF reveals AI generator — critical clue",
                "ai_probability": 0.88,
                "verdict": "EXIF_REVEALS_AI_GENERATOR",
            },
            "reverse_image_search": {
                "earliest_known_date": None,
                "diffusion_model_fingerprint": True,
                "ai_generated_probability": 0.94,
                "verdict": "DIFFUSION_MODEL_GENERATED_FACE",
            },
        },
    },

    # ─────────────────────────────────────────────────────
    # TASK 5 — Human-authored content (ground_truth=False)
    # Tests false-positive avoidance — the cardinal-sin penalty.
    # An agent that blindly submits is_ai_generated=True scores 0.00.
    # ─────────────────────────────────────────────────────
    "task_human": {
        "case_id": "case_005",
        "content_type": "text",
        "content_data": (
            "Internal legal memo — Q3 Compliance Review (CONFIDENTIAL).\n"
            "Author: J. Harrington, General Counsel. Date: 2024-11-14.\n\n"
            "Following the Meridian acquisition close, several legacy data-"
            "retention policies remain misaligned with GDPR Art. 17. The "
            "affected datasets — primarily Meridian's pre-2021 CRM exports "
            "stored on their Frankfurt cluster — must be reviewed before "
            "31 Dec. I've flagged this to Priya's team (ref: TKT-4492) but "
            "haven't had a response. Recommend we escalate via the DPA "
            "liaison before the regulatory window closes.\n\n"
            "Side note: the grammar inconsistencies in sections 3-4 are from "
            "J.H.'s dictation software, not a drafting error."
        ),
        "ground_truth": False,
        "critical_evidence": ["text_perplexity_scan"],
        # Relevant for 'text': analyze_metadata, text_perplexity_scan, check_domain_age
        # reverse_image_search is NOT relevant for text — removed from tool_responses
        # so agents who try it receive no evidence AND get -0.20 penalty (consistent).
        "tool_responses": {
            "text_perplexity_scan": {
                "perplexity_score": 91.7,
                "ai_probability": 0.14,
                "verdict": "LIKELY_HUMAN_WRITTEN",
                "note": (
                    "High perplexity consistent with informal human dictation. "
                    "Internal jargon (TKT-4492, Meridian acquisition), "
                    "named individuals, and grammar inconsistencies all reduce "
                    "AI probability significantly."
                ),
            },
            "analyze_metadata": {
                "author": "J. Harrington",
                "created": "2024-11-14T09:22:11Z",
                "last_modified_by": "J. Harrington",
                "revisions": 3,
                "suspicious": False,
                "note": "Clean metadata — consistent with claimed author and timeline.",
            },
            "check_domain_age": {
                "verdict": "NOT_APPLICABLE",
                "note": "Internal document — no external domain to check.",
            },
        },
    },
}
