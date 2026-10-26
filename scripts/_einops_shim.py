"""einops 的纯 torch 替代实现（不安装任何包）。

为什么需要
----------
`layers/SelfAttention_Family.py:7` 有 `from einops import rearrange, repeat`，
但本机三个 python 环境都没有 einops。用户约束是「不 pip install」，
所以这里实现一个够用的子集。

覆盖范围
--------
只实现上游代码实际用到的模式（都是"符号命名 + 括号分组"的重排）：

    rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
    rearrange(x, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
    repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model',
           repeat=batch)

解析规则（与 einops 一致的部分）：
  * 左侧每个空格分隔的 token 对应输入的一个轴；带括号的 token 表示若干子轴合并
  * 右侧同理，对输出轴分组
  * 出现两次以上的轴名（如 `ts_d` 左右都有）按语义对齐
  * `repeat=` 传入的轴可以出现在右侧（用于 repeat）

不支持：省略号 `...`、`reduce`、`parse_shape`。
若遇到不支持的模式会明确抛 NotImplementedError，不会静默给错结果。
"""

from __future__ import annotations

import sys
import types

__all__ = ["rearrange", "repeat", "reduce"]


def _split_groups(expr: str):
    """'b (h w) c' -> ['b', ['h','w'], 'c']"""
    out, buf, depth = [], "", 0
    for ch in expr.strip():
        if ch == "(":
            depth += 1
            buf += ch
        elif ch == ")":
            depth -= 1
            buf += ch
        elif ch == " " and depth == 0:
            if buf:
                out.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    groups = []
    for g in out:
        if g.startswith("(") and g.endswith(")"):
            groups.append([t for t in g[1:-1].split() if t])
        else:
            groups.append([g])
    return groups


def _flat(groups):
    names = []
    for g in groups:
        names.extend(g)
    return names


def rearrange(x, pattern: str, **axes):
    if "..." in pattern or "->" not in pattern:
        raise NotImplementedError(f"不支持的 einops 模式: {pattern}")
    lhs, rhs = pattern.split("->")
    lg, rg = _split_groups(lhs), _split_groups(rhs)

    lnames = _flat(lg)
    shape = list(x.shape)
    if len(shape) != len(lg):
        raise ValueError(f"输入维度 {len(shape)} 与模式左侧 {len(lg)} 轴不匹配: {pattern}")

    # 已知轴长
    dim = {}
    for i, g in enumerate(lg):
        if len(g) == 1:
            dim[g[0]] = shape[i]
    for k, v in axes.items():
        dim[k] = int(v)

    # 推断被括号合并的轴（长度未知时均分/用已知轴求）
    for i, g in enumerate(lg):
        if len(g) == 1:
            continue
        known = [dim[n] for n in g if n in dim]
        unknown = [n for n in g if n not in dim]
        if unknown:
            prod_known = 1
            for v in known:
                prod_known *= v
            rem = shape[i] // max(prod_known, 1)
            if len(unknown) == 1:
                dim[unknown[0]] = rem
            else:
                # 无信息则从左到右均分，最后一个兜底
                acc = 1
                for n in unknown[:-1]:
                    guess = max(rem // (len(unknown)), 1)
                    dim[n] = guess
                    acc *= guess
                dim[unknown[-1]] = rem // max(acc, 1)

    # 展开为逐轴
    target_in = []
    for g in lg:
        for n in g:
            target_in.append(dim[n])
    if target_in != shape:
        raise ValueError(f"推断的输入形状 {target_in} 与实际 {shape} 不符: {pattern}")

    # 先展平到"平铺轴"
    flat_names = lnames
    perm_flat = list(range(len(flat_names)))

    # 输出需要的轴顺序（按右侧顺序，逐轴）
    out_names = []
    for g in rg:
        for n in g:
            if n in axes and n not in flat_names:
                # repeat 轴：需要先 unsqueeze 再 expand
                out_names.append(n)
            elif n in flat_names:
                out_names.append(n)
            else:
                raise ValueError(f"输出轴 {n} 不在输入中且未通过 kwargs 提供: {pattern}")

    y = x
    # 展开括号轴
    new_shape = []
    for g in lg:
        new_shape.extend([dim[n] for n in g])
    y = y.reshape(new_shape)

    # repeat 引入的新轴
    extra = []
    for g in rg:
        for n in g:
            if n not in flat_names:
                extra.append(n)
    for n in extra:
        y = y.unsqueeze(0)
        flat_names = [n] + flat_names
    if extra:
        expand_shape = [dim[n] if n in extra else y.shape[i + len(extra)]
                        for i, n in enumerate(flat_names)]
        y = y.expand(*expand_shape)

    # 按输出顺序 permute
    order = [flat_names.index(n) for n in out_names]
    y = y.permute(*order)

    # 合并输出分组
    if any(len(g) > 1 for g in rg):
        final = []
        for g in rg:
            p = 1
            for n in g:
                p *= dim[n] if n in dim else y.shape[out_names.index(n)]
            final.append(p)
        y = y.reshape(final)
    return y


def repeat(x, pattern: str, **axes):
    return rearrange(x, pattern, **axes)


def reduce(x, pattern: str, reduction: str = "mean", **axes):  # noqa: A002
    raise NotImplementedError("本项目未使用 einops.reduce")


def install():
    """若真 einops 缺失，注册 shim 模块。"""
    if "einops" in sys.modules:
        return
    try:
        import einops  # noqa: F401
        return
    except ImportError:
        pass
    m = types.ModuleType("einops")
    m.rearrange = rearrange
    m.repeat = repeat
    m.reduce = reduce
    sys.modules["einops"] = m
