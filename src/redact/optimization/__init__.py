"""General optimize-toward-an-objective search (own top-level folder).

A beam/tree search over conversation *moves* (the controller above
``llms.conversation.drive_generators``): expand candidate next-turns in parallel,
score each with a judge, keep the best beam, log the whole tree. Objective-general —
``multiturn_attacks`` uses it to optimize jailbreaks, but any setting with a scorer
can. See ``.claude/theme4_multiturn_plan.md`` (`optimization/`, weak-point W1).
"""

from .search import Node, optimize, tree_to_records

__all__ = ["Node", "optimize", "tree_to_records"]
