# CerebroCast 实验套件

论文《基于因果脑机制的普适持续预测架构》（工作代号 CerebroCast）的全部实验代码、
结果、统计与图表。**所有图由本目录下的 matplotlib 代码从结果文件直接绘制，
不使用任何生成式图像模型，因此每张图都可逐字节复现。**

> 结果与结论见 [`results/summary/report.md`](results/summary/report.md)。
> 该报告如实记录了**未达标**的预注册假设，不作任何事后挑选。

## 1. 运行环境

```bash
# 本机环境（Python 3.13 隔离环境，numpy / scipy / pandas / scikit-learn /
# matplotlib / statsmodels）
C:/Users/Administrator/.workbuddy/binaries/python/envs/stog/Scripts/python.exe
```

`C:\code\T` 中的既有环境因 Python 3.15 就地升级导致 numpy / torch 的 C 扩展
（cp314 二进制）与解释器不匹配，已弃用；本套件不依赖 GPU，全流程 CPU 可复现。

**外部数据依赖**：`src/cerebrocast/data.py` 中的 `DATA_ROOT` 指向
`C:\paper-STOG\Experiments\dataset`（上一稿实验所用数据集：EPF 五个分区、
ETT h1/h2/m1/m2、Weather、ECL）。这些数据集均为公开数据集，可按同名目录结构放置。

## 2. 目录

```
实验/
  src/cerebrocast/     架构与实验代码
    config.py          全部超参、消融臂定义、预注册阈值
    data.py            11 个域的加载、日切分、时序划分
    experts.py         专家池（12 通用 + 9 异质：3 训练时期 + 6 regime 专精）
    fingerprints.py    模块(1) 谱指纹编码器
    memory.py          模块(2) 情景记忆 + 模块(3) 慢亲和图 / 簇场
    system.py          模块(4)-(9) 与完整日循环（Algorithm 1）
    baselines.py       对照组 B1-B7
    protocol.py        与臂无关的 winsorising 包络（主指标）
    combo.py           单纯形约束最小二乘（含收缩与加权）
    synth.py           受控漂移合成生成器（真值已知）
    stats.py           块自助、TOST 等效检验、BH 校正、宏 F1、恢复时间
  scripts/
    build_pool.py      Stage 1  拟合专家池并缓存逐日预测（增长式外样本协议）
    build_stress.py    Stage 1b 构建非平稳阶梯 STAT-0/1、SW-2/4/X
    run_stream.py      Stage 2  日历流式主实验（校准 → 部署）
    run_synth.py       Stage 3  受控漂移实验（H2 / H4 / P2）
    run_stats.py       Stage 4  统计与预注册判定表
    make_figures.py    Stage 5  全部图表（PDF / SVG / 300 dpi PNG）
    diag_*.py          方法调试诊断脚本（非交付路径，保留以备审查）
  results/ pool / stream / synth / summary
  figures/
```

## 3. 复现流程

```bash
PY="C:/Users/Administrator/.workbuddy/binaries/python/envs/stog/Scripts/python.exe"
$PY scripts/build_pool.py            # ~25 min
$PY scripts/build_stress.py 7 42 2021 3407 12345   # ~1 min
$PY scripts/run_stream.py --rounds 2  # ~50 min
$PY scripts/run_synth.py
$PY scripts/run_stats.py
$PY scripts/make_figures.py
```

`results/pool` 与 `results/stream` 中的 `.npz` 是中间产物，体积较大，不随仓库分发；
按上面的顺序重跑即可完整重建。

## 4. 协议要点

* **日历流式重放**：第 t 日系统仅可见 ≤ t−1 日数据；日末真实值到达后计算损失并
  运行睡眠相；测试段内永不整体重训（重训基线除外）。
* **增长式外样本专家池**（本次修订）：整条序列切成
  `fit 55% | warm 10% | val 10% | test 25%` 四段，专家池随数据累积重拟合两次，
  于是架构所见到的**每一天**都由从未见过该天的模型预测：
  warm 段用 fit 段拟合的池，val 段用 fit+warm 拟合的池，test 段用 fit+warm+val 的池。
  fit 段自身的预测不入库（那是样本内的）。
* **标定**：θ₁、θ₂、η、γ、资格迹半衰期、top-k 等标量在验证段一次性网格标定后
  冻结，此后不再接触测试段。
* **主指标**：与臂无关的 winsorised 日均损失
  `W = mean_t min(loss_t, cap_t)`，`cap_t` 只由专家池当日损失分布的 95 分位×8
  经 EMA 平滑得到，任何臂都不能通过自我截断获益。
* **配对设计**：专家池只做一次实现（seed 7），5 个种子只改变架构自身的随机性，
  因此架构与全部对照在同一批日子、同一个池上逐日配对比较。

## 5. 关于专家池的说明

专家池刻意保持**异质**：12 个在同一拟合段上训练的通用模型，3 个只用了拟合段
25% / 50% / 75% 前缀的"旧版本"模型，以及 9 个 regime 专精模型（3 个簇 × Ridge / HGB）。
理由：若专家池同质，最优*固定*组合已接近最优，持续维护问题在数学上是退化的，
任何维护方法都不可能有收益——这是实验可被证伪的前提，而非调参技巧。真实部署的池
同样是异质的（不同时期上线、不同数据段训练的模型并存）。

### 5.1 已修正的一处严重缺陷（务必保留记录）

上一版池采用"在训练段一次性拟合、再预测整条序列"的做法。检查发现距离加权
k-NN 专家（`knn_lag`）在这种做法下会**自匹配**：查询点本身就在邻居集合中，距离为 0，
`weights="distance"` 于是把它自己的训练目标原样复制回去。实测在 EPF-NP 训练段
**1523 天中有 1003 天（66%）损失恰为 0**，而测试段为 0/437。这使池看起来远比实际
强大，并污染了架构的预热信号（它会过度信任一个"完美"的专家）。
改用上面的增长式外样本协议后该现象自动消失（同一专家外样本 MSE 由 0.0000 变为 0.2322）。
