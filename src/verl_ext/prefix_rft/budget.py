"""Turn a prefix fraction into a token split over a multi-decision demonstration.

The paper's prefix is a token fraction of one response (A.2). A demonstration here
is ``m`` decisions, so the fraction is taken over their concatenation: whole
decisions are replayed while they fit the budget, and the one that straddles it is
split.

``budget <= total - 1`` is the paper's ``prefix_len >= demo_len -> demo_len - 1``
guard (recorded in the 2026-08-17 spec against ``recipe/prefix_rft/rl_dataset.py:300-301``)
applied to the concatenation. It guarantees at least one generated token, and with it
``n_full < len(lengths)`` whenever ``r > 0``, so the split decision always exists.

No tokenizer and no verl: this is arithmetic, and keeping it separate is what lets it
be tested in the agent_engine env.
"""

from __future__ import annotations

import math


def split_for_budget(lengths, l):
    """Return ``(n_full, r)`` for prefix fraction ``l`` over decision token ``lengths``.

    ``n_full`` decisions are replayed whole, then ``r`` tokens of decision
    ``n_full + 1``. ``r == 0`` means the budget landed on a decision boundary and
    nothing is split, which is exactly step mode at ``k = n_full``.
    """
    total = sum(lengths)
    if total <= 1:
        # Nothing can be replayed without consuming the only token there is.
        return 0, 0

    budget = int(math.floor(l * total))
    budget = max(0, min(budget, total - 1))

    n_full = 0
    used = 0
    for n in lengths:
        if used + n > budget:
            break
        used += n
        n_full += 1
    return n_full, budget - used
