# zkPMT 实验代码

论文 **"Zero-Knowledge Proof-Based Integrity Verification for DNN Training (zkPMT)"** 的实验复现代码，包含主实验和消融实验。

---

## 目录结构

```
实验/
├── main.py               # 主实验入口
├── models.py             # 模型定义 (MLP, ResNet50)
├── fixed_point.py        # 定点编码 + Poseidon 哈希模拟
├── zkpmt.py              # zkPMT 协议完整实现
├── baselines.py          # 5 个对比方案
├── data_loader.py        # 数据集加载
├── experiments.py        # 主实验运行器
├── plot_results.py       # 主实验绘图（Fig. 4–8）
├── requirements.txt
├── README.md
│
└── ablation/             # 消融实验（独立子文件夹）
    ├── main_ablation.py      # 消融实验入口
    ├── ablation_variants.py  # 6 个消融变体定义
    ├── run_ablation.py       # 消融实验运行器
    ├── plot_ablation.py      # 消融实验绘图
    ├── results/              # 实验结果（自动生成）
    └── figures/              # 图表输出（自动生成）
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 完整实验流程

### 第一步：主实验（对比 6 个方案）

```bash
python main.py
```

运行四个任务（Small+Iris、Medium+MNIST、Large+CIFAR-10、ResNet50+CIFAR-10），对比 zkPMT 与 Garg、Kaizen、zkCNN、zkDL、VeriCNN，生成论文 Fig. 4–8。

### 第二步：消融实验

```bash
cd ablation
python main_ablation.py
```

逐一移除 zkPMT 的四个关键组件，量化每个组件的贡献，生成消融实验图表。

---

## 所有运行模式

### 主实验 `main.py`

| 命令                         | 说明                       | 耗时   |
| ---------------------------- | -------------------------- | ------ |
| `python main.py`             | 完整实验（3次重复，100轮） | 数小时 |
| `python main.py --fast`      | 快速验证（1次重复，20轮）  | 数分钟 |
| `python main.py --plot-only` | 仅重新绘图（需已有结果）   | 秒级   |

### 消融实验 `ablation/main_ablation.py`

| 命令                                  | 说明                     | 耗时   |
| ------------------------------------- | ------------------------ | ------ |
| `python main_ablation.py`             | 完整消融实验             | 数小时 |
| `python main_ablation.py --fast`      | 快速验证                 | 数分钟 |
| `python main_ablation.py --plot-only` | 仅重新绘图               | 秒级   |

---

## 输出文件

### 主实验输出（`实验/` 目录下）

| 文件                            | 内容                       |
| ------------------------------- | -------------------------- |
| `results/results.json`          | 原始实验数据               |
| `figures/fig4_init_cost.png`    | 初始化开销（论文 Fig. 4）  |
| `figures/fig5_single_prove.png` | 单轮证明生成开销（Fig. 5） |
| `figures/fig6_total_prove.png`  | 累计证明生成开销（Fig. 6） |
| `figures/fig7_verify.png`       | 验证开销（Fig. 7）         |
| `figures/fig8_storage.png`      | 存储开销（Fig. 8）         |
| `figures/fig6c_scalability.png` | 可扩展性折线图             |

### 消融实验输出（`ablation/` 目录下）

| 文件                                     | 内容                         |
| ---------------------------------------- | ---------------------------- |
| `results/ablation_results.json`          | 消融实验原始数据             |
| `figures/ablation_fig1_init.png`         | 各变体初始化开销             |
| `figures/ablation_fig2_prove.png`        | 各变体证明生成开销           |
| `figures/ablation_fig3_verify.png`       | 各变体验证开销               |
| `figures/ablation_fig4_storage.png`      | 各变体存储开销               |
| `figures/ablation_fig5_gates.png`        | 各变体电路门数量             |
| `figures/ablation_fig6_contribution.png` | 各组件贡献度（相对开销增量） |

---

## 消融实验设计

zkPMT 的四个关键技术组件：

| 组件                | 说明                                   | 移除后的影响                              |
| ------------------- | -------------------------------------- | ----------------------------------------- |
| **A. 统一定点编码** | 互补码映射 + 延迟缩放，减少缩放/舍入门 | 电路门数增加约 2.5×，初始化和证明开销上升 |
| **B. 电路复用**     | 一次编译，后续轮次仅更新 witness       | 每轮重新编译，初始化变为 O(T·g)           |
| **C. 状态哈希链**   | Poseidon 哈希绑定相邻轮次模型状态      | 无法检测跳轮或中间状态篡改                |
| **D. 递归聚合**     | 每 10 轮聚合一次，单次验证全程         | 验证开销变为 O(T)，存储增加               |

六个消融变体：

| 变体               | A   | B   | C   | D   |
| ------------------ | --- | --- | --- | --- |
| `zkPMT-Full`       | ✓   | ✓   | ✓   | ✓   |
| `w/o-FixedPoint`   | ✗   | ✓   | ✓   | ✓   |
| `w/o-CircuitReuse` | ✓   | ✗   | ✓   | ✓   |
| `w/o-HashChain`    | ✓   | ✓   | ✗   | ✓   |
| `w/o-Recursion`    | ✓   | ✓   | ✓   | ✗   |
| `w/o-FP+Reuse`     | ✗   | ✗   | ✓   | ✓   |

---

## 注意事项

- ZKP 操作通过**校准时序模型**模拟，基于论文报告的实测数据。真实 ZKP 实现需要 Rust + Bellman 库。
- MNIST 和 CIFAR-10 首次运行需联网下载，失败时自动切换为同维度合成数据。
- 若结果文件不存在，绘图模块将使用空占位结构并生成空柱图，便于保留代码流程。
