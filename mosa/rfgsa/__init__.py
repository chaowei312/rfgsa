"""Router-Free Gated Sparse Attention (RFGSA) package.

Modules:
    load_balance    — RoutingFreeLoadBalance (adaptive L_EB + L_TB)
    core            — PureRFGSA (norm scoring, sigmoid gate, ExpertScatter)
    linear_gate     — PureRFGSA_LinearGate (simple linear scoring)
    softmax_gate    — PureRFGSA_SoftmaxGate (linear + softmax gating)
                      PureRFGSA_NormSoftmaxGate (norm + softmax gating)
    concat          — PureRFGSA_Concat (slice concat + shared W_o)
    kv_cache        — SparseKVCache + rfgsa_ar_step (AR with eviction)
    hybrid          — RFGSA (sparse + dense/local wrapper)
"""

from .load_balance import RoutingFreeLoadBalance
from .core import PureRFGSA
from .linear_gate import PureRFGSA_LinearGate
from .softmax_gate import PureRFGSA_SoftmaxGate, PureRFGSA_NormSoftmaxGate
from .concat import PureRFGSA_Concat
from .kv_cache import SparseKVCache, rfgsa_ar_step
from .hybrid import RFGSA
