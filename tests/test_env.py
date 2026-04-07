"""
tests/test_env.py — Digital Forensics Lab Environment Tests

Covers:
  - OpenEnv contract: reset / step / state
  - Typed model validation (ForensicAction, ForensicReward)
  - Reward ranges on all action types
  - Irrelevant tools: penalised AND do not store evidence
  - Unknown actions: penalised AND deduct budget
  - Grader correctness across all scoring tiers
  - False positive / false negative paths
  - Proportional critical-evidence scoring
  - Budget depletion (TOOL_COST = 0.15)
  - Episode boundary enforcement
  - All tasks initialise and run clean
  - task_medium (audio) and task_hard (hybrid) specific paths

Run with:
    pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import (
    DigitalForensicsEnv,
    ForensicAction,
    ForensicReward,
    CaseFileState,
    ActionResult,
    VALID_TASKS,
    ALL_TOOLS,
    TOOL_COST,
)


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = DigitalForensicsEnv("task_easy")
    env.reset()
    return env


@pytest.fixture
def medium_env():
    env = DigitalForensicsEnv("task_medium")
    env.reset()
    return env


@pytest.fixture
def hard_env():
    env = DigitalForensicsEnv("task_hard")
    env.reset()
    return env


@pytest.fixture
def human_env():
    env = DigitalForensicsEnv("task_human")
    env.reset()
    return env


@pytest.fixture
def expert_env():
    env = DigitalForensicsEnv("task_expert")
    env.reset()
    return env


# ─────────────────────────────────────────────────────
# 1. Typed model presence (OpenEnv spec)
# ─────────────────────────────────────────────────────

def test_forensic_action_is_pydantic_model():
    """ForensicAction must be a Pydantic BaseModel with action + params fields."""
    a = ForensicAction(action="analyze_metadata")
    assert a.action == "analyze_metadata"
    assert a.params == {}


def test_forensic_action_with_params():
    """ForensicAction carries verdict params correctly."""
    a = ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.9, "critical_evidence_keys": []},
    )
    assert a.params["is_ai_generated"] is True


def test_forensic_reward_is_pydantic_model():
    """ForensicReward must be a Pydantic BaseModel with value + rationale."""
    r = ForensicReward(value=0.50, rationale="Critical tool used.")
    assert r.value == 0.50
    assert isinstance(r.rationale, str)


def test_action_result_contains_reward_detail(easy_env):
    """ActionResult must expose typed reward_detail (ForensicReward)."""
    result = easy_env.step(ForensicAction(action="analyze_metadata"))
    assert isinstance(result.reward_detail, ForensicReward)
    assert result.reward_detail.value == result.reward
    assert len(result.reward_detail.rationale) > 0


def test_step_accepts_forensic_action_model(easy_env):
    """step() must accept a ForensicAction model (not just raw string)."""
    result = easy_env.step(ForensicAction(action="run_ast_code_analyzer"))
    assert result.reward == 0.50  # critical tool


# ─────────────────────────────────────────────────────
# 2. Spec compliance — reset / state / step contract
# ─────────────────────────────────────────────────────

def test_valid_tasks_exist():
    assert len(VALID_TASKS) == 5


def test_valid_tasks_names():
    expected = {"task_easy", "task_medium", "task_hard", "task_expert", "task_human"}
    assert set(VALID_TASKS) == expected


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        DigitalForensicsEnv("task_does_not_exist")


@pytest.mark.parametrize("task", VALID_TASKS)
def test_reset_returns_clean_state(task):
    env = DigitalForensicsEnv(task)
    obs = env.reset()
    assert obs.budget_remaining == 1.0
    assert obs.step_count == 0
    assert obs.done is False
    assert obs.gathered_evidence == {}
    assert obs.last_action_error is None
    assert obs.case_id != ""
    assert obs.content_type != ""


@pytest.mark.parametrize("task", VALID_TASKS)
def test_state_matches_last_obs(task):
    env = DigitalForensicsEnv(task)
    obs = env.reset()
    state = env.state()
    assert state.case_id == obs.case_id
    assert state.budget_remaining == obs.budget_remaining
    assert state.step_count == obs.step_count


def test_step_before_reset_raises():
    env = DigitalForensicsEnv("task_easy")
    with pytest.raises(RuntimeError, match="reset\\(\\)"):
        env.step("analyze_metadata")


def test_step_after_done_raises(easy_env):
    easy_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.9,
                "critical_evidence_keys": ["run_ast_code_analyzer"]},
    ))
    with pytest.raises(RuntimeError, match="Episode is over"):
        easy_env.step("analyze_metadata")


# ─────────────────────────────────────────────────────
# 3. Reward ranges
# ─────────────────────────────────────────────────────

@pytest.mark.parametrize("task", VALID_TASKS)
def test_task_score_in_range(task):
    env = DigitalForensicsEnv(task)
    env.reset()
    result = env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.5, "critical_evidence_keys": []},
    ))
    score = result.info.get("task_score", -1)
    assert 0.0 <= score <= 1.0, f"task_score {score} out of range for {task}"


@pytest.mark.parametrize("task", VALID_TASKS)
def test_step_rewards_in_range(task):
    env = DigitalForensicsEnv(task)
    env.reset()
    for tool in ALL_TOOLS[:3]:
        result = env.step(tool)
        assert -1.0 <= result.reward <= 1.0


# ─────────────────────────────────────────────────────
# 4. Tool reward logic
# ─────────────────────────────────────────────────────

def test_critical_tool_gives_high_reward(easy_env):
    result = easy_env.step("run_ast_code_analyzer")
    assert result.reward == 0.50


def test_relevant_noncritical_tool_gives_positive_reward(easy_env):
    result = easy_env.step("analyze_metadata")
    assert result.reward == 0.20


def test_irrelevant_tool_gives_penalty(easy_env):
    # task_easy is 'code'; audio scan is irrelevant
    result = easy_env.step("run_audio_frequency_scan")
    assert result.reward == -0.20


def test_irrelevant_tool_does_not_store_evidence(easy_env):
    """Irrelevant tools must NOT add to gathered_evidence."""
    easy_env.step("run_audio_frequency_scan")  # irrelevant for code
    assert "run_audio_frequency_scan" not in easy_env.state().gathered_evidence


def test_relevant_tool_stores_evidence(easy_env):
    """Relevant tools MUST add to gathered_evidence."""
    easy_env.step("run_ast_code_analyzer")
    assert "run_ast_code_analyzer" in easy_env.state().gathered_evidence


def test_duplicate_tool_gives_penalty(easy_env):
    easy_env.step("analyze_metadata")
    result = easy_env.step("analyze_metadata")
    assert result.reward == -0.40


def test_budget_decreases_by_tool_cost(easy_env):
    """Each tool call deducts exactly TOOL_COST from budget."""
    before = easy_env.state().budget_remaining
    easy_env.step("analyze_metadata")
    after = easy_env.state().budget_remaining
    assert round(before - after, 4) == TOOL_COST


def test_unknown_action_penalised_and_deducts_budget(easy_env):
    """Unknown action: -0.20 reward AND budget deducted (no free looping)."""
    before = easy_env.state().budget_remaining
    result = easy_env.step("nonexistent_tool")
    after = easy_env.state().budget_remaining
    assert result.reward == -0.20
    assert result.observation.last_action_error is not None
    assert round(before - after, 4) == TOOL_COST  # budget was deducted


# ─────────────────────────────────────────────────────
# 5. Grader — correct verdict scoring tiers
# ─────────────────────────────────────────────────────

def test_correct_verdict_all_critical_scores_1(easy_env):
    easy_env.step("run_ast_code_analyzer")
    result = easy_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.92,
                "critical_evidence_keys": ["run_ast_code_analyzer"]},
    ))
    assert result.info["task_score"] == 0.99
    assert result.info["result"] == "CORRECT_ALL_CRITICAL_EVIDENCE"


def test_correct_verdict_no_evidence_scores_half(easy_env):
    result = easy_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.5,
                "critical_evidence_keys": []},
    ))
    assert result.info["task_score"] == 0.50
    assert result.info["result"] == "CORRECT_NO_EVIDENCE"


def test_correct_verdict_noncritical_evidence_scores_0_60(easy_env):
    easy_env.step("analyze_metadata")  # relevant but not critical
    result = easy_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.7,
                "critical_evidence_keys": ["analyze_metadata"]},
    ))
    assert result.info["task_score"] == 0.60
    assert result.info["result"] == "CORRECT_WITH_PARTIAL_EVIDENCE"


# ─────────────────────────────────────────────────────
# 6. Grader — false positive (cardinal sin) and false negative
# ─────────────────────────────────────────────────────

def test_false_positive_scores_zero(human_env):
    result = human_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.9,
                "critical_evidence_keys": []},
    ))
    assert result.info["task_score"] == 0.01
    assert result.info["result"] == "FALSE_POSITIVE"


def test_correct_not_ai_with_critical_evidence_scores_1(human_env):
    human_env.step("text_perplexity_scan")
    result = human_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": False, "confidence": 0.85,
                "critical_evidence_keys": ["text_perplexity_scan"]},
    ))
    assert result.info["task_score"] == 0.99


def test_false_negative_scores_0_10(easy_env):
    result = easy_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": False, "confidence": 0.6,
                "critical_evidence_keys": []},
    ))
    assert result.info["task_score"] == 0.10
    assert result.info["result"] == "FALSE_NEGATIVE"


# ─────────────────────────────────────────────────────
# 7. Proportional scoring — task_expert (2 critical tools)
# ─────────────────────────────────────────────────────

def test_expert_partial_critical_scores_between_50_and_75(expert_env):
    """1 of 2 critical tools → score between 0.50 and 0.75."""
    expert_env.step("analyze_metadata")
    result = expert_env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.7,
                "critical_evidence_keys": ["analyze_metadata"]},
    ))
    score = result.info["task_score"]
    # At 1/2 critical ratio (0.50), formula gives 0.75 exactly — that's the lower
    # bound of the MAJORITY tier, so <= 0.75 is correct here.
    assert 0.50 < score <= 0.75, f"Expected partial score, got {score}"
    assert result.info["critical_cited_count"] == 1
    assert result.info["critical_total_count"] == 2


def test_expert_all_critical_scores_1():
    env = DigitalForensicsEnv("task_expert")
    env.reset()
    env.step("analyze_metadata")
    env.step("reverse_image_search")
    result = env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.95,
                "critical_evidence_keys": ["analyze_metadata", "reverse_image_search"]},
    ))
    assert result.info["task_score"] == 0.99


# ─────────────────────────────────────────────────────
# 8. task_medium (audio) specific paths
# ─────────────────────────────────────────────────────

def test_medium_audio_scan_is_critical(medium_env):
    """run_audio_frequency_scan is the critical tool for task_medium."""
    result = medium_env.step("run_audio_frequency_scan")
    assert result.reward == 0.50
    assert "run_audio_frequency_scan" in medium_env.state().gathered_evidence


def test_medium_caller_id_trap(medium_env):
    """analyze_metadata on audio returns clean caller_id — not the critical tool."""
    medium_env.step("analyze_metadata")
    evidence = medium_env.state().gathered_evidence
    assert "analyze_metadata" in evidence
    assert evidence["analyze_metadata"]["suspicious"] is False  # THE TRAP


def test_medium_full_perfect_episode():
    """Full perfect episode for task_medium → task_score == 1.00."""
    env = DigitalForensicsEnv("task_medium")
    env.reset()
    env.step("run_audio_frequency_scan")
    result = env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.97,
                "critical_evidence_keys": ["run_audio_frequency_scan"]},
    ))
    assert result.info["task_score"] == 0.99


# ─────────────────────────────────────────────────────
# 9. task_hard (hybrid) specific paths
# ─────────────────────────────────────────────────────

def test_hard_text_trap_does_not_conclude_human(hard_env):
    """text_perplexity_scan on task_hard returns LIKELY_HUMAN_WRITTEN — the trap."""
    hard_env.step("text_perplexity_scan")
    evidence = hard_env.state().gathered_evidence
    assert evidence["text_perplexity_scan"]["verdict"] == "LIKELY_HUMAN_WRITTEN"


def test_hard_image_and_video_are_critical(hard_env):
    """Both reverse_image_search and run_temporal_consistency_check are critical."""
    r1 = hard_env.step("reverse_image_search")
    r2 = hard_env.step("run_temporal_consistency_check")
    assert r1.reward == 0.50
    assert r2.reward == 0.50


def test_hard_ast_analyzer_has_response(hard_env):
    """run_ast_code_analyzer is relevant for hybrid and must return a response."""
    hard_env.step("run_ast_code_analyzer")
    evidence = hard_env.state().gathered_evidence
    assert "run_ast_code_analyzer" in evidence
    assert evidence["run_ast_code_analyzer"]["verdict"] == "NOT_APPLICABLE"


def test_hard_full_perfect_episode():
    """Full perfect episode for task_hard → task_score == 1.00."""
    env = DigitalForensicsEnv("task_hard")
    env.reset()
    env.step("reverse_image_search")
    env.step("run_temporal_consistency_check")
    result = env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.95,
                "critical_evidence_keys": [
                    "reverse_image_search", "run_temporal_consistency_check"
                ]},
    ))
    assert result.info["task_score"] == 0.99


# ─────────────────────────────────────────────────────
# 10. Budget depletion (TOOL_COST = 0.15)
# ─────────────────────────────────────────────────────

def test_budget_depletion_ends_episode():
    """
    Exhausting budget before verdict ends episode with reward=-1.00.
    With TOOL_COST=0.15 and budget=1.0, the 7th tool call pushes budget
    to -0.05 (<=0), triggering forced episode end.
    """
    env = DigitalForensicsEnv("task_easy")
    env.reset()
    result = None
    # Run distinct tools until budget depletes.
    # 6 distinct tools = 0.90 used. 7th would be a duplicate (only 3 relevant
    # for 'code') — either way budget goes <= 0 and episode ends.
    for tool in ALL_TOOLS:
        if env.state().done:
            break
        result = env.step(tool)
    # If not done yet, keep using duplicates to force depletion
    while not env.state().done:
        result = env.step("analyze_metadata")

    assert result is not None
    assert result.done is True
    assert result.reward == -1.00


def test_budget_exactly_six_tools_safe():
    """
    Running exactly 6 distinct relevant-or-irrelevant tools should leave
    budget at 1.0 - 6*0.15 = 0.10 (positive). Agent can still submit.
    """
    env = DigitalForensicsEnv("task_hard")  # hybrid: all 7 tools relevant
    env.reset()
    # Run 6 of the 7 tools
    for tool in ALL_TOOLS[:6]:
        env.step(tool)
    assert env.state().budget_remaining > 0
    assert not env.state().done
    # Can still submit verdict
    result = env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.9,
                "critical_evidence_keys": [
                    "reverse_image_search", "run_temporal_consistency_check"
                ]},
    ))
    assert result.done is True


# ─────────────────────────────────────────────────────
# 11. Episode reset is idempotent
# ─────────────────────────────────────────────────────

def test_reset_clears_previous_episode():
    env = DigitalForensicsEnv("task_easy")
    env.reset()
    env.step("run_ast_code_analyzer")
    env.step(ForensicAction(
        action="submit_verdict",
        params={"is_ai_generated": True, "confidence": 0.9,
                "critical_evidence_keys": ["run_ast_code_analyzer"]},
    ))
    assert env.state().done is True

    obs = env.reset()
    assert obs.done is False
    assert obs.budget_remaining == 1.0
    assert obs.gathered_evidence == {}
    assert obs.step_count == 0


# ─────────────────────────────────────────────────────
# 12. Evidence accumulation
# ─────────────────────────────────────────────────────

def test_relevant_tool_responses_stored(easy_env):
    easy_env.step("analyze_metadata")
    easy_env.step("run_ast_code_analyzer")
    evidence = easy_env.state().gathered_evidence
    assert "analyze_metadata" in evidence
    assert "run_ast_code_analyzer" in evidence
    assert isinstance(evidence["run_ast_code_analyzer"], dict)


def test_task_human_reverse_image_search_not_relevant():
    """
    reverse_image_search is NOT relevant for text content.
    It must be penalised and must NOT appear in gathered_evidence.
    """
    env = DigitalForensicsEnv("task_human")
    env.reset()
    result = env.step("reverse_image_search")
    assert result.reward == -0.20
    assert "reverse_image_search" not in env.state().gathered_evidence
