"""`keep` selects the solve mode, and ALL / [] are opposites.

Spelling "fully symbolic" as None made None and [] two *falsy* values with
opposite meanings, so `keep or ()` silently merged them across the codebase.
These lock the three consequences, none of which announced itself:

  * the solve cache handed a NUMERIC result back for a SYMBOLIC request
  * Result could not record which of the two it held
  * the cost estimator costed the cheapest case when asked about the one whose
    cost is unbounded -- so the machinery meant to warn you was blind to the
    only case worth warning about
"""
import warnings
from pathlib import Path

import pytest

from circuitinsight import ALL, SessionController, SolveTooLarge, is_all
from circuitinsight.keep import norm_keep

FIX = Path(__file__).resolve().parent / "fixtures" / "spectre" / "ota5t"


@pytest.fixture(scope="module")
def sess():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield SessionController.open(str(FIX / "tb_ota5t.cin.json"),
                                     str(FIX / "psf"))


# ------------------------------------------------------------------ sentinel

def test_all_is_not_truth_testable():
    """`if keep:` is exactly the bug; make it impossible rather than discouraged."""
    with pytest.raises(TypeError, match="no truth value"):
        bool(ALL)


def test_is_all_distinguishes_the_opposites():
    assert is_all(ALL)
    assert not is_all(None)                      # None now aliases [] ...
    assert not is_all([])                        # ... the OPPOSITE of ALL
    assert not is_all(["MN0"])


def test_norm_keep_gives_distinct_cache_keys():
    """ALL and [] must never share a key; None and [] must ALWAYS."""
    assert norm_keep(ALL) != norm_keep([])
    assert norm_keep(None) == norm_keep([]) == ()
    assert norm_keep(["b", "a"]) == ("b", "a")   # order preserved, hashable


def test_keep_none_is_numeric(sess):
    """keep=None means keep=[] everywhere: same result, same cache entry,
    and never the unbounded symbolic solve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_none = sess.solve("VIND", "vout", None)
        r_empty = sess.solve("VIND", "vout", [])
    assert r_none is r_empty                     # one cache entry
    assert r_none.keep == []


# ------------------------------------------------------- estimator sees ALL

def test_estimate_costs_the_symbolic_case(sess):
    """estimate_solve_time(ALL) used to be coerced to [] -- i.e. it reported the
    cheapest solve when asked about the unbounded one."""
    an = sess._analyzer_ready()
    numeric = an.estimate_solve_time("VIND", "vout", [])
    symbolic = an.estimate_solve_time("VIND", "vout", ALL)
    assert symbolic.grid_size > numeric.grid_size
    assert symbolic.grid_size > 10_000          # every symbol of a 13-device ckt


# --------------------------------------------------------------- the guard

def test_unbounded_solve_is_refused_not_hung(sess):
    """A fully symbolic solve does not terminate (the paper says so, §IV). Refuse
    it -- with the way out in the message -- rather than hang the caller."""
    with pytest.raises(SolveTooLarge) as ei:
        sess.solve("VIND", "vout", ALL, max_seconds=1.0)
    msg = str(ei.value)
    assert "fully symbolic" in msg and "plan_keep" in msg


def test_keep_all_refused_even_with_no_budget(sess):
    """max_seconds=None means 'no cap', not 'run the impossible'. keep=ALL cannot
    finish, so there is no sense in which the caller wanted it to proceed."""
    with pytest.raises(SolveTooLarge):
        sess.solve("VIND", "vout", ALL, max_seconds=None)


def test_a_slow_hybrid_solve_is_NOT_refused_by_default(sess):
    """The guard must not block the tool's headline result.

    The paper's flagship is a hybrid solve keeping all twelve conductances of the
    two-stage amp: ~250 s. A default budget (I first set 60 s) would have REFUSED
    it. Slow-but-finite is the user's call; only the non-terminating case is ours.
    """
    an = sess._analyzer_ready()
    keep = [d.name for d in sess.devices if d.device_type == "mosfet"][:4]
    est = an.estimate_solve_time("VIND", "vout", keep)

    # Whatever it costs, no budget was requested -> the guard must stay out of it.
    sess._guard_cost("VIND", "vout", keep, None)          # must not raise

    # And with an explicit budget below the estimate, it *does* speak up.
    if est.seconds is not None and est.seconds > 0.01:
        with pytest.raises(SolveTooLarge):
            sess._guard_cost("VIND", "vout", keep, est.seconds / 100.0)


def test_numeric_solve_is_not_refused(sess):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sess.solve("VIND", "vout", [])
    assert r.dc_gain_db == pytest.approx(46.13, abs=0.1)
    assert r.keep == []                          # numeric, and it says so


# ------------------------------------- keep names that match nothing raise

def test_wrong_form_keep_name_raises(sess):
    """keep=["MN0.gm"] (instance.param) used to bake the symbol numerically
    with no error: the miss was invisible until you inspected free_symbols,
    since TransferFunction.symbols still listed every system symbol. The
    correct spelling is the symbol name, param_instance ("gm_MN0")."""
    from circuitinsight.engine.mna import MnaError, hybrid_split

    system = sess._analyzer_ready().system("VIND")
    good = next(n for n in system.symbols if n.startswith("gm_"))
    inst = good[len("gm_"):]
    wrong = f"{inst}.gm"                         # the tempting wrong order

    with pytest.raises(MnaError) as ei:
        hybrid_split(system, [wrong])
    msg = str(ei.value)
    assert wrong in msg                          # names the offender
    assert "param_instance" in msg               # and the convention
    assert good in msg                           # and the close match

    # one bad entry poisons the set even when the others are fine
    with pytest.raises(MnaError):
        hybrid_split(system, [good, wrong])


def test_valid_keep_forms_still_match(sess):
    """Exact symbol name and bare instance name (suffix match) both keep."""
    from circuitinsight.engine.mna import hybrid_split

    system = sess._analyzer_ready().system("VIND")
    good = next(n for n in system.symbols if n.startswith("gm_"))
    inst = good[len("gm_"):]

    _, kept = hybrid_split(system, [good])       # exact name
    assert kept == [good]
    _, kept = hybrid_split(system, [inst])       # instance keeps all its symbols
    assert good in kept


def test_solve_surfaces_the_bad_keep_name(sess):
    """The error must reach tf(), not just direct hybrid_split callers."""
    from circuitinsight.engine.mna import MnaError

    with pytest.raises(MnaError, match="matched no symbol"):
        sess._analyzer_ready().tf("VIND", "vout", keep=["MN0.gm"])


# ------------------------------------------- the cache no longer conflates

def test_numeric_result_is_not_served_for_a_symbolic_request(sess):
    """The sharp end of the collision: solve numerically, then ask for symbolic.

    With `tuple(keep or ())` both hashed to (), so the second call was served the
    cached NUMERIC result -- silently answering a different question. Now the
    keys differ, so the symbolic request is a genuine (and here, refused) solve.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numeric = sess.solve("VIND", "vout", [])
    assert numeric.keep == []

    with pytest.raises(SolveTooLarge):           # NOT the cached numeric Result
        sess.solve("VIND", "vout", ALL, max_seconds=1.0)


def test_result_records_which_mode_produced_it(sess):
    """Result.keep: ALL = fully symbolic, [] = numeric. list[str] could not
    represent the former, so it was coerced to [] and the summary mislabelled it."""
    from circuitinsight.gui import view

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sess.solve("VIND", "vout", [])
    assert r.keep == [] and r.keep is not None
    assert "numeric" in view.summary_text(r).splitlines()[0]
