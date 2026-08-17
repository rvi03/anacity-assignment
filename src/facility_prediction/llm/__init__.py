"""The LLM track's package.

Modules here import the shared pipeline. Targets, splits, tolerances,
and metrics are defined there and are never redefined in this package;
a second definition would make the two tracks' numbers incomparable.
"""

from __future__ import annotations

TRACK = "llm"

# The state carried until a run either completes or stops at a gate.
BRANCH_NOT_STARTED = "not_started"
