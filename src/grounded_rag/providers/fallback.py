"""An ordered chain of providers with a per-provider circuit breaker.

This is failover between different services, not a retry of the same call: when the
first provider answers with an error (rate limit, outage, truncated reply), the next
one gets the same prompt. A provider that fails ``failure_threshold`` times in a row
is skipped for ``cooldown`` seconds, so a dead service does not add its timeout to
every question, and one probe is allowed once the cooldown is over. When every
provider is failing or cooling down, the chain fails fast with one message that names
each of them, and the pipeline degrades to its extractive answer.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from grounded_rag.envelopes import ModelTier
from grounded_rag.providers.base import BackendError, BackendUnavailable, Completer

DEFAULT_THRESHOLD = 2
DEFAULT_COOLDOWN = 60.0


@dataclass
class _Circuit:
    consecutive_failures: int = 0
    open_until: float = 0.0
    last_error: str = ""


class FallbackBackend:
    def __init__(
        self,
        members: Sequence[tuple[str, Completer]],
        *,
        failure_threshold: int = DEFAULT_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN,
        clock: Callable[[], float] = time.monotonic,
        skipped: Mapping[str, str] | None = None,
    ) -> None:
        if not members:
            raise ValueError("a fallback chain needs at least one provider")
        self.members = list(members)
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.clock = clock
        self.skipped = dict(skipped or {})
        self.name = "chain(" + ">".join(name for name, _ in self.members) + ")"
        self.last_provider: str | None = None
        self._circuits = {name: _Circuit() for name, _ in self.members}

    def complete(self, prompt: str, *, system: str, tier: ModelTier = "small") -> str:
        problems: list[str] = []
        for name, backend in self.members:
            circuit = self._circuits[name]
            now = self.clock()
            if circuit.open_until > now:
                remaining = circuit.open_until - now
                problems.append(f"{name}: cooling down {remaining:.0f}s after {circuit.last_error}")
                continue
            try:
                text = backend.complete(prompt, system=system, tier=tier)
            except BackendError as exc:
                circuit.consecutive_failures += 1
                circuit.last_error = str(exc)
                if circuit.consecutive_failures >= self.failure_threshold:
                    circuit.open_until = now + self.cooldown
                problems.append(f"{name}: {exc}")
                continue
            circuit.consecutive_failures = 0
            circuit.open_until = 0.0
            self.last_provider = name
            return text
        raise BackendError("all providers failed: " + "; ".join(problems))

    def status(self) -> dict[str, dict[str, object]]:
        now = self.clock()
        return {
            name: {
                "state": "cooling_down" if circuit.open_until > now else "ready",
                "consecutive_failures": circuit.consecutive_failures,
                "last_error": circuit.last_error,
            }
            for name, circuit in self._circuits.items()
        }


def build_chain(
    names: Sequence[str],
    env: Mapping[str, str] | None = None,
    **options: Any,
) -> FallbackBackend:
    """Build a chain in the given order, leaving out providers that cannot be used here.

    A provider with no credentials or no SDK installed is recorded in ``skipped`` with the
    reason, so the same command works on a machine that only has some of them. An unknown
    name is a typo, not an environment difference: ``backend_from_env`` raises ``ValueError``
    and it is not swallowed.
    """
    from grounded_rag.providers import backend_from_env  # the registry lives in the package

    members: list[tuple[str, Completer]] = []
    skipped: dict[str, str] = {}
    for name in names:
        try:
            members.append((name, backend_from_env(name, env)))
        except BackendUnavailable as exc:
            skipped[name] = str(exc)
    if not members:
        reasons = "; ".join(skipped.values())
        raise BackendUnavailable(f"no provider in the chain can be used: {reasons}")
    return FallbackBackend(members, skipped=skipped, **options)
