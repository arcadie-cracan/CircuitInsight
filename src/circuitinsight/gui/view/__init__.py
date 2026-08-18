"""Pure presentation helpers: turn a `Result` into a Matplotlib figure and
display strings. No Qt, no ipywidgets — both front ends (and the tests) reuse
this with the plain Agg backend.

Split into format / simplify / figures / report; this __init__ re-exports
every name so ``from . import view`` callers are untouched.
"""
from .format import (  # noqa: F401
    SIG,
    _ENG_PREFIX_TEX,
    _FACTORS_PER_LINE,
    _GREEK,
    _QTY,
    _REGIONS,
    _SPECIAL,
    _eng_coeff_tex,
    _eng_tex,
    _factor_tex,
    _fmt_root,
    _inst_sub,
    _pair_roots,
    _product_tex,
    _tok,
    _wrapped_product,
    latex_eng,
    mirror_map,
    op_unit,
    poles_table,
    ranking_rows,
    region_name,
    symbol_tex,
)
from .simplify import (  # noqa: F401
    _best_grouping,
    _dominant_scale,
    _drop_common_scale,
    _edge_ratio,
    _expr_lines,
    _fold_unit_floats,
    _num_den,
    _raw_tf_lines,
    _rgcd,
    expr_katex,
    expr_value_map,
    numeral_tips,
    prepare_display,
    round_expr,
    tf_latex,
)
from .figures import (  # noqa: F401
    _OVERLAY_COLORS,
    _annotate_margins,
    _fig_b64,
    _legend_anchor_y,
    _pole_zero_ticks,
    bode_figure,
    error_figure,
    expr_figure,
    fidelity,
    figure_legend,
    refresh_legend,
    whatif_fn,
)
from .report import (  # noqa: F401
    html_report,
    markdown_report,
    report_section,
    session_report,
    summary_text,
    traces_csv,
)
from ...units import eng  # noqa: F401  (re-exported: view.eng)
__all__ = ["bode_figure", "summary_text", "poles_table", "eng"]
