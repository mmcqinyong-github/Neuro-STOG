"""依赖 shim：为本机缺失的 `reformer_pytorch` 提供一个占位模块。

为什么要这样
------------
`layers/SelfAttention_Family.py:6` 顶层 `from reformer_pytorch import LSHSelfAttention`，
但 `LSHSelfAttention` 只在该文件第 221 行的 ReformerLayer 里用到。
我们只用 FullAttention，完全不触 Reformer。

不改上游源码（保持零改写），改为在 import 之前把假模块塞进 `sys.modules`。
若真的调用到 LSH，立刻抛错——不会静默给出错误结果。

同理补齐 `einops` 若缺失（本机有）。
"""

from __future__ import annotations

import sys
import types


def install():
    _install_reformer()
    _install_einops()


def _install_reformer():
    if "reformer_pytorch" in sys.modules:
        return
    try:
        import reformer_pytorch  # noqa: F401
        return
    except ImportError:
        pass

    m = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention:                              # noqa: D101
        def __init__(self, *a, **kw):
            raise NotImplementedError(
                "LSHSelfAttention 是给 Reformer 用的占位；本项目只用 FullAttention。"
                "若走到这里，说明误用了 Reformer 架构。")

    m.LSHSelfAttention = LSHSelfAttention
    sys.modules["reformer_pytorch"] = m


def _install_einops():
    if "einops" in sys.modules:
        return
    try:
        import einops  # noqa: F401
        return
    except ImportError:
        pass
    import importlib.util
    from pathlib import Path
    p = Path(__file__).with_name("_einops_shim.py")
    spec = importlib.util.spec_from_file_location("_einops_shim", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install()
