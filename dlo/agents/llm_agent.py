"""LLM decision agent: one call per review day, up to 2 validation retries (charged), then hold."""
from __future__ import annotations

from dataclasses import dataclass, field

from dlo import config
from dlo.agents import prompts
from dlo.agents.schema import Decision, ValidationFailed, decision_json_schema, validate
from dlo.clients.openrouter_chat import BudgetExceeded, ChatRequest, LLMClient, LLMError


@dataclass
class AgentResult:
    decision: Decision | None
    status: str                      # valid_first | valid_retry | invalid | error
    attempts: int
    cost: float
    call_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0


async def decide(client: LLMClient, model_key: str, user_msg: str, traj_id: str, sim_day: str,
                 budget: str) -> AgentResult:
    messages = [{"role": "system", "content": prompts.decision_system()}, {"role": "user", "content": user_msg}]
    max_retries = config.costs()["max_retries"]
    res = AgentResult(decision=None, status="invalid", attempts=0, cost=0.0)
    for attempt in range(max_retries + 1):
        res.attempts = attempt + 1
        try:
            out = await client.chat(ChatRequest(
                model_key=model_key, messages=list(messages), purpose="decision" if attempt == 0 else "decision_retry",
                budget=budget, charged=True, trajectory_id=traj_id, sim_day=sim_day, attempt=attempt,
                json_schema=decision_json_schema()))
        except BudgetExceeded:
            raise
        except LLMError as e:
            res.errors.append(f"api: {e}")
            res.status = "error"
            return res
        res.cost += out.cost
        res.call_ids.append(out.call_id)
        res.reasoning_tokens += out.reasoning_tokens
        res.completion_tokens += out.completion_tokens
        res.prompt_tokens += out.prompt_tokens
        try:
            res.decision = validate(out.content)
            res.status = "valid_first" if attempt == 0 else "valid_retry"
            return res
        except ValidationFailed as e:
            res.errors.append(str(e))
            messages += [{"role": "assistant", "content": out.content},
                         {"role": "user", "content": f"Your answer failed validation: {e}. "
                                                     "Return the full corrected JSON decision."}]
    res.status = "invalid"
    return res
