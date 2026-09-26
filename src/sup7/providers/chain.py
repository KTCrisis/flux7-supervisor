"""Provider chain — several evaluators tried in order, with a circuit breaker.

The first provider that answers gives the verdict; an "escalate" verdict is an
answer, not a failure. The next provider is tried only when the previous one
fails (returns None: network error, HTTP error such as 402, timeout,
unreadable answer). If every provider fails, the chain returns None and the
rule evaluator escalates to a human, as with a single provider.

A provider that fails `failures` times in a row is skipped for `cooldown`
seconds, so a dead or unpaid provider does not add its latency to every
approval. The verdict's reasoning is prefixed with the provider that answered
and the ones skipped or failed, so the trace says who decided.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from sup7.models import ApprovalContext, Verdict

from .base import Evaluator

logger = logging.getLogger(__name__)


class ChainEvaluator:
    def __init__(
        self,
        providers: list[tuple[str, Evaluator]],
        failures: int = 3,
        cooldown: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("provider chain is empty")
        self._providers = providers
        self._max_failures = max(1, failures)
        self._cooldown = cooldown
        self._clock = clock
        self._failures = [0] * len(providers)
        self._open_until = [0.0] * len(providers)

    async def evaluate(self, approval: ApprovalContext) -> Verdict | None:
        passed: list[str] = []  # providers skipped or failed before the answer
        for i, (name, provider) in enumerate(self._providers):
            now = self._clock()
            if self._open_until[i] > now:
                passed.append(f"{name} skipped")
                continue
            try:
                verdict = await provider.evaluate(approval)
            except Exception as e:  # a provider must not break the chain
                logger.warning("provider %s raised: %s", name, e)
                verdict = None
            if verdict is None:
                self._failures[i] += 1
                if self._failures[i] >= self._max_failures:
                    self._open_until[i] = now + self._cooldown
                    self._failures[i] = 0
                    logger.warning(
                        "provider %s failed %d times, skipped for %.0fs",
                        name, self._max_failures, self._cooldown,
                    )
                passed.append(f"{name} failed")
                continue
            self._failures[i] = 0
            prefix = f"[{name}" + (f", {', '.join(passed)}" if passed else "") + "] "
            return Verdict(verdict.action, verdict.confidence, prefix + verdict.reasoning)
        logger.warning("every provider in the chain failed: %s", ", ".join(passed))
        return None

    async def close(self) -> None:
        for _, provider in self._providers:
            await provider.close()
