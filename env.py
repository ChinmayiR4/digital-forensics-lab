"""
env.py - Digital Forensics Lab Core Environment

Implements the OpenEnv interface:
  reset()       -> CaseFileState  (initial observation)
  step(action)  -> ActionResult   (observation, reward, done, info)
  state()       -> CaseFileState  (current state, read-only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TYPED MODELS (OpenEnv spec — all three required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ForensicAction   typed Action   — pass to step()
  CaseFileState    typed Observation — returned by reset()/step()
  ForensicReward   typed Reward   — embedded in ActionResult

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD DESIGN  (two distinct layers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 1 — Step Rewards  [-1.0 to +1.0]
  +0.50   CRITICAL tool used (key evidence for this case)
  +0.20   RELEVANT tool used (valid for content type, not critical)
  -0.20   IRRELEVANT tool used (wrong for content type)
           Evidence is NOT stored for irrelevant tools — penalty
           and grader are consistent.
  -0.20   UNKNOWN action (budget still deducted — no free looping)
  -0.40   DUPLICATE tool (already ran this tool)
  -1.00   Budget depleted before verdict

LAYER 2 — Task Score  [0.0 to 1.0]
  0.99       Correct verdict + ALL critical evidence cited
  0.75-0.99   Correct verdict + >=50% critical evidence (scaled)
  0.50-0.74   Correct verdict + <50% critical evidence (scaled)
  0.60        Correct verdict + only non-critical evidence
  0.50        Correct verdict, zero evidence (lucky guess)
  0.10        False negative (missed real AI content)
  0.05       False positive (accused human of being AI — cardinal sin)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUDGET DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOOL_COST = 0.15 per call.
  Budget starts at 1.0.
  Running all 7 tools costs 1.05 — more than the budget.
  Effective maximum before forced depletion: 6 tools.
  Agents must choose which tools to run.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from mock_database import CASES


# ─────────────────────────────────────────────────────
# Typed Action Model  (OpenEnv spec)
# ─────────────────────────────────────────────────────

class ForensicAction(BaseModel):
    """
    Typed Action model — pass to step() for spec-compliant usage.

    Fields:
        action  Tool name (one of ALL_TOOLS) or 'submit_verdict'
        params  Required only for submit_verdict:
                {
                  "is_ai_generated": bool,
                  "confidence": float (0.0-1.0),
                  "critical_evidence_keys": list[str]
                }
                Empty dict for all tool actions.
    """
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────
# Typed Reward Model  (OpenEnv spec)
# ─────────────────────────────────────────────────────

class ForensicReward(BaseModel):
    """
    Typed Reward model — embedded in ActionResult.reward_detail.

    Fields:
        value      Reward in [-1.0, +1.0] for steps;
                   task_score in [0.0, 1.0] on verdict step
        rationale  Human-readable explanation of the reward given
    """
    value: float
    rationale: str


# ─────────────────────────────────────────────────────
# Typed Observation Model  (OpenEnv spec)
# ─────────────────────────────────────────────────────

class CaseFileState(BaseModel):
    """
    Observation space — everything the agent sees at each step().

    Fields:
        case_id           Unique case identifier (e.g. 'case_001')
        content_type      'code' | 'audio' | 'image' | 'video' | 'text' | 'hybrid'
        content_data      The suspicious content (text representation)
        budget_remaining  Starts at 1.0, decreases 0.15 per tool used
        gathered_evidence tool_name -> JSON output (relevant tools only)
        step_count        Number of actions taken so far
        done              True when episode ended
        last_action_error Error from last action, or None
    """
    case_id: str
    content_type: str
    content_data: str
    budget_remaining: float
    gathered_evidence: Dict[str, Any]
    step_count: int
    done: bool
    last_action_error: Optional[str] = None


# ─────────────────────────────────────────────────────
# ActionResult
# ─────────────────────────────────────────────────────

class ActionResult(BaseModel):
    """
    Return type of step().

    Fields:
        observation    Updated CaseFileState
        reward         Step reward as float [-1.0, +1.0]  (backward compat)
        reward_detail  Typed ForensicReward with value + rationale
        done           True if episode ended
        info           Extra context; includes task_score on final step
    """
    observation: CaseFileState
    reward: float
    reward_detail: ForensicReward
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────────────
# Tool Relevance Map
# ─────────────────────────────────────────────────────

RELEVANT_TOOLS: Dict[str, List[str]] = {
    "code":   ["analyze_metadata", "run_ast_code_analyzer", "text_perplexity_scan"],
    "audio":  ["analyze_metadata", "run_audio_frequency_scan", "text_perplexity_scan", "check_domain_age"],
    "image":  ["analyze_metadata", "reverse_image_search"],
    "video":  ["analyze_metadata", "run_temporal_consistency_check", "run_audio_frequency_scan"],
    "text":   ["analyze_metadata", "text_perplexity_scan", "check_domain_age"],
    "hybrid": [
        "analyze_metadata", "reverse_image_search", "text_perplexity_scan",
        "check_domain_age", "run_audio_frequency_scan",
        "run_ast_code_analyzer", "run_temporal_consistency_check",
    ],
}

ALL_TOOLS = [
    "analyze_metadata",
    "reverse_image_search",
    "text_perplexity_scan",
    "check_domain_age",
    "run_audio_frequency_scan",
    "run_ast_code_analyzer",
    "run_temporal_consistency_check",
]

VALID_TASKS = list(CASES.keys())

# 0.15 per tool: running all 7 costs 1.05 > 1.0 budget.
# Max safe tool calls before forced depletion: 6.
TOOL_COST = 0.15


# ─────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────

class DigitalForensicsEnv:
    """
    Digital Forensics Lab — OpenEnv Environment.

    Usage (typed, recommended):
        env = DigitalForensicsEnv("task_easy")
        obs = env.reset()
        result = env.step(ForensicAction(action="run_ast_code_analyzer"))
        result = env.step(ForensicAction(
            action="submit_verdict",
            params={"is_ai_generated": True, "confidence": 0.92,
                    "critical_evidence_keys": ["run_ast_code_analyzer"]}
        ))

    Usage (string shorthand, backward compatible):
        result = env.step("run_ast_code_analyzer")
        result = env.step("submit_verdict", {"is_ai_generated": True, ...})
    """

    def __init__(self, task: str) -> None:
        if task not in CASES:
            raise ValueError(
                f"Unknown task '{task}'.\n"
                f"Valid tasks: {VALID_TASKS}"
            )
        self.task = task
        self.case_data = CASES[task]
        self._state: Optional[CaseFileState] = None
        self._tools_used: set = set()

    # ── Public OpenEnv Interface ──────────────────────

    def reset(self) -> CaseFileState:
        """Reset environment and return initial observation."""
        self._tools_used = set()
        self._state = CaseFileState(
            case_id=self.case_data["case_id"],
            content_type=self.case_data["content_type"],
            content_data=self.case_data["content_data"],
            budget_remaining=1.0,
            gathered_evidence={},
            step_count=0,
            done=False,
            last_action_error=None,
        )
        return self._state

    def step(
        self,
        action: Union[str, ForensicAction],
        params: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """
        Take one action in the environment.

        Args:
            action: ForensicAction model (preferred) OR action string.
            params: Only used when action is a plain string.

        Returns:
            ActionResult with observation, reward, reward_detail, done, info.
        """
        # Normalise input to (action_str, params_dict)
        if isinstance(action, ForensicAction):
            params = action.params
            action_str = action.action
        else:
            action_str = str(action)
            if params is None:
                params = {}

        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is over. Call reset() to start again.")

        self._state.step_count += 1
        reward = 0.0
        rationale = ""
        done = False
        info: Dict[str, Any] = {}
        error: Optional[str] = None

        if action_str == "submit_verdict":
            done = True
            self._state.done = True
            reward, info = self._grade_verdict(params)
            rationale = info.get("feedback", "Verdict submitted.")

        elif action_str in ALL_TOOLS:
            reward, error, rationale = self._use_tool(action_str)

        else:
            # Unknown action: penalise AND deduct budget.
            # Without budget deduction an agent could loop on bad actions forever.
            self._state.budget_remaining = round(
                self._state.budget_remaining - TOOL_COST, 4
            )
            reward = -0.20
            rationale = f"Unknown action '{action_str}' — budget deducted."
            error = (
                f"Unknown action '{action_str}'. "
                f"Valid: {ALL_TOOLS + ['submit_verdict']}"
            )

        # Budget exhausted before verdict → forced episode end, heavy penalty
        if self._state.budget_remaining <= 0 and not done:
            done = True
            self._state.done = True
            reward = -1.00
            rationale = "Budget exhausted before submitting verdict."
            info["budget_depleted"] = True
            info["message"] = "Budget exhausted before submitting verdict."

        self._state.last_action_error = error

        return ActionResult(
            observation=self._state,
            reward=round(reward, 2),
            reward_detail=ForensicReward(
                value=round(reward, 2),
                rationale=rationale,
            ),
            done=done,
            info=info,
        )

    def state(self) -> CaseFileState:
        """Return current state without advancing the episode."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ── Private Helpers ───────────────────────────────

    def _use_tool(self, tool_name: str) -> Tuple[float, Optional[str], str]:
        """
        Execute a forensic tool. Returns (reward, error_or_None, rationale).

        CRITICAL RULE: Evidence is stored in gathered_evidence ONLY when the
        tool is relevant for the content_type. Irrelevant tools are penalised
        AND produce no evidence. The penalty and the grader are now consistent.
        """
        content_type = self.case_data["content_type"]
        relevant = RELEVANT_TOOLS.get(content_type, ALL_TOOLS)

        # Duplicate penalty (applied before relevance check)
        if tool_name in self._tools_used:
            self._state.budget_remaining = round(
                self._state.budget_remaining - TOOL_COST, 4
            )
            return (
                -0.40,
                f"Tool '{tool_name}' already used — running it again wastes budget.",
                f"Duplicate '{tool_name}' — no new evidence, budget wasted.",
            )

        self._tools_used.add(tool_name)
        self._state.budget_remaining = round(
            self._state.budget_remaining - TOOL_COST, 4
        )

        # Irrelevant tool: penalise and do NOT store evidence
        if tool_name not in relevant:
            return (
                -0.20,
                (
                    f"Tool '{tool_name}' is not relevant for {content_type} content. "
                    f"Relevant tools: {relevant}"
                ),
                (
                    f"'{tool_name}' is irrelevant for {content_type} — "
                    f"no evidence stored, budget deducted."
                ),
            )

        # Relevant tool: store evidence if a mock response exists
        tool_responses = self.case_data["tool_responses"]
        if tool_name in tool_responses:
            self._state.gathered_evidence[tool_name] = tool_responses[tool_name]

        critical = self.case_data["critical_evidence"]

        if tool_name in critical:
            return (
                +0.50,
                None,
                f"'{tool_name}' is a CRITICAL tool for this case — key evidence found.",
            )

        return (
            +0.20,
            None,
            f"'{tool_name}' is relevant but not the critical tool for this case.",
        )

    def _grade_verdict(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Grade submit_verdict. Returns (task_score, info_dict).
        Uses proportional critical-evidence ratio.
        """
        ground_truth: bool = self.case_data["ground_truth"]
        critical_evidence: List[str] = self.case_data["critical_evidence"]

        is_ai_generated: bool = bool(params.get("is_ai_generated", False))
        confidence: float = float(params.get("confidence", 0.5))
        cited_keys: List[str] = params.get("critical_evidence_keys", [])

        verdict_correct = (is_ai_generated == ground_truth)

        gathered_keys = list(self._state.gathered_evidence.keys())
        n_critical = len(critical_evidence)
        n_cited_critical = sum(
            1 for k in critical_evidence
            if k in cited_keys and k in gathered_keys
        )
        critical_ratio = n_cited_critical / n_critical if n_critical > 0 else 0.0

        has_any_evidence = (
            len(cited_keys) > 0
            and any(k in gathered_keys for k in cited_keys)
        )

        info = {
            "ground_truth": ground_truth,
            "submitted_verdict": is_ai_generated,
            "confidence": confidence,
            "cited_evidence": cited_keys,
            "critical_evidence_needed": critical_evidence,
            "gathered_evidence_keys": gathered_keys,
            "critical_evidence_ratio": round(critical_ratio, 2),
            "critical_cited_count": n_cited_critical,
            "critical_total_count": n_critical,
        }

        if verdict_correct:
            if critical_ratio == 1.0:
                task_score = 1.00
                info["result"] = "CORRECT_ALL_CRITICAL_EVIDENCE"
                info["feedback"] = (
                    f"Perfect. Cited all {n_critical} critical evidence "
                    f"piece(s) and reached the correct verdict."
                )
            elif critical_ratio >= 0.5:
                task_score = round(0.75 + (critical_ratio - 0.5) * 0.50, 2)
                info["result"] = "CORRECT_MAJORITY_CRITICAL_EVIDENCE"
                info["feedback"] = (
                    f"Good. Cited {n_cited_critical}/{n_critical} critical pieces. "
                    f"Find the remaining {n_critical - n_cited_critical} for 1.00."
                )
            elif critical_ratio > 0.0:
                task_score = round(0.50 + critical_ratio * 0.50, 2)
                info["result"] = "CORRECT_SOME_CRITICAL_EVIDENCE"
                info["feedback"] = (
                    f"Correct verdict but only {n_cited_critical}/{n_critical} "
                    f"critical evidence cited. Deeper investigation required."
                )
            elif has_any_evidence:
                task_score = 0.60
                info["result"] = "CORRECT_WITH_PARTIAL_EVIDENCE"
                info["feedback"] = (
                    "Correct verdict with supporting evidence, but the most "
                    "important forensic tools were not used or cited."
                )
            else:
                task_score = 0.50
                info["result"] = "CORRECT_NO_EVIDENCE"
                info["feedback"] = (
                    "Correct verdict but no evidence cited. Lucky guess."
                )
        else:
            if is_ai_generated and not ground_truth:
                task_score = 0.00
                info["result"] = "FALSE_POSITIVE"
                info["feedback"] = (
                    "You accused a human of being AI — the cardinal sin "
                    "in Trust & Safety. Never flag clean content."
                )
            else:
                task_score = 0.10
                info["result"] = "FALSE_NEGATIVE"
                info["feedback"] = (
                    "You missed real AI-generated content. "
                    "The disinformation got through."
                )

        # Clamp to strictly (0, 1) — validator rejects 0.0 and 1.0
        task_score = round(min(max(task_score, 0.01), 0.99), 2)
        info["task_score"] = task_score
        return task_score, info
