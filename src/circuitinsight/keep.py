"""The `keep` sentinel: which device symbols stay symbolic in a solve.

`keep` selects the solve mode, and two of its values are *opposites*:

    ALL             fully symbolic -- every device symbol retained. The point
                    of the tool, and the only mode whose cost is unbounded: on
                    a large circuit it can run for hours. Ask
                    estimate_solve_time() first.
    [] (or None)    fully numeric -- no symbols retained. Cheap.
    ["M1", "CL"]    hybrid -- only these stay symbolic.

History: "fully symbolic" was once spelled `None`, which made `None` and `[]`
two *falsy* values with OPPOSITE meanings; every `keep or ()` silently merged
them (the solve cache handed numeric results back for symbolic requests, the
cost estimator costed the cheap case when asked about the unbounded one).
`ALL` is now the ONLY spelling of fully-symbolic -- explicit, and it refuses
truth-testing -- while `None` is a harmless alias of `[]`: an absent keep set
means "nothing kept". Use `is_all(keep)` / `norm_keep(keep)`.
"""
from __future__ import annotations

__all__ = ["ALL", "is_all", "norm_keep"]


class _KeepAll:
    """Sentinel: retain every device symbol (fully symbolic solve)."""

    __slots__ = ()

    def __repr__(self) -> str:                  # pragma: no cover - cosmetic
        return "ALL"

    def __bool__(self) -> bool:
        raise TypeError(
            "keep=ALL has no truth value -- `if keep:` would conflate ALL "
            "(fully symbolic) with [] (fully numeric), which are opposites. "
            "Use `is_all(keep)`.")


ALL = _KeepAll()


def is_all(keep) -> bool:
    """True when `keep` asks for a fully symbolic solve -- spelled ALL,
    explicitly. None is NOT symbolic: it aliases [] (numeric)."""
    return keep is ALL


def norm_keep(keep):
    """Normalize to ALL or a tuple of names — a hashable, unambiguous key.
    None normalizes to () -- the numeric solve."""
    if is_all(keep):
        return ALL
    return () if keep is None else tuple(keep)
