"""Impact-ionization (substrate-current) conductances in the MOSFET model.

Spectre's ac/stb linearization includes dI_sub/dv terms that dcOpInfo never
reports (r2r_bias_pos_loop investigation): a drain->bulk current controlled
by v(d,s) and v(b,s). `gii_d`/`gii_m` expose them as first-class MOSFET
params — positive coefficients for both polarities in the (d, b, ., s)
orientation, symbols gii_d_<inst>/gii_m_<inst> like gm — and absent params
must emit nothing, leaving every ordinary circuit untouched.
"""
from circuitinsight.adapters.cin import FlatDevice
from circuitinsight.models.small_signal import expand_device

_OP = {"gm": 6.98e-5, "gmbs": 2.57e-5, "gds": 4.81e-8,
       "cgs": 2.3e-14, "cgd": 4.4e-15, "cgb": 6.8e-15,
       "cdb": 8.6e-15, "csb": 1.9e-14, "cjd": 8.6e-15, "cjs": 1.3e-14}
_K = {f"k{i}{j}": 1e-15 for i in "dgb" for j in "dgb"}


def _mos(params=None):
    return FlatDevice(name="MN0", sim_name="MN0", device_type="mosfet",
                      terminals={"d": "vdn", "g": "vgn", "s": "vsn", "b": "0"},
                      params=params or {})


def _gii_prims(prims):
    return {p.param: p for p in prims if p.param.startswith("gii")}


def test_gii_params_expand_to_drain_bulk_vccs_pair():
    dev = _mos({"gii_d": "4.0e-8", "gii_m": "2.7e-7"})
    for cap_model, op in (("lumped", _OP), ("matrix", {**_OP, **_K})):
        gii = _gii_prims(expand_device(dev, op, cap_model=cap_model))
        assert set(gii) == {"gii_d", "gii_m"}, cap_model
        assert gii["gii_d"].kind == "vccs"
        assert gii["gii_d"].nodes == ("vdn", "0", "vdn", "vsn")   # i(d->b) ~ v(d,s)
        assert gii["gii_m"].nodes == ("vdn", "0", "0", "vsn")     # i(d->b) ~ v(b,s)
        assert gii["gii_d"].value == 4.0e-8
        assert gii["gii_m"].value == 2.7e-7


def test_absent_gii_params_emit_nothing():
    for cap_model, op in (("lumped", _OP), ("matrix", {**_OP, **_K})):
        prims = expand_device(_mos(), op, cap_model=cap_model)
        assert not _gii_prims(prims), cap_model
