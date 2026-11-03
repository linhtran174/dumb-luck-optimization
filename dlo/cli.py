"""dlo command line."""
from __future__ import annotations

import asyncio
import json

import typer

app = typer.Typer(help="Dumb Luck Optimization benchmark", no_args_is_help=True)


@app.command()
def smoke() -> None:
    """M1: one structured call to each pinned model; prints cost, provider, reasoning tokens."""
    from dlo.clients.openrouter_chat import ChatRequest, OpenRouterClient
    from dlo.store import LLMLedger

    schema = {"name": "smoke", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["answer", "confidence"],
        "properties": {"answer": {"type": "integer"}, "confidence": {"type": "number"}}}}

    async def run():
        client = OpenRouterClient(LLMLedger())
        try:
            for key in ("terra", "luna", "deepseek"):
                res = await client.chat(ChatRequest(
                    model_key=key, purpose="smoke", budget="infra",
                    messages=[{"role": "user", "content": "What is 17 * 23? Reply in JSON."}],
                    json_schema=schema, max_tokens=2000))
                parsed = json.loads(res.content)
                typer.echo(json.dumps({"model": key, "response_model": res.response_model, "provider": res.provider,
                                       "parsed": parsed, "cost": res.cost, "reasoning_tokens": res.reasoning_tokens,
                                       "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                                       "cached": res.cached}))
        finally:
            await client.aclose()

    asyncio.run(run())


@app.command()
def ledger() -> None:
    """Spend summary from the persisted LLM ledger."""
    from dlo.store import LLMLedger

    led = LLMLedger()
    for r in led.summary():
        typer.echo(json.dumps(r))
    for b in ("trial", "dev", "test", "infra"):
        typer.echo(f"{b}: ${led.spent(b):.4f}")


def _register() -> None:
    # Subcommand groups are added by later milestones; imported lazily to keep `dlo --help` fast.
    from dlo import commands  # noqa: F401


try:
    _register()
except ImportError:
    pass

if __name__ == "__main__":
    app()
