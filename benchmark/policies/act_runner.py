"""ACT runner placeholder.

The benchmark interface is fixed here first. Connect real ACT training and
inference code after the 50 recorded episodes are available.
"""

from __future__ import annotations

from benchmark.policies.base_policy import NotConnectedPolicy


class ACTRunner(NotConnectedPolicy):
    name = "ACT"
