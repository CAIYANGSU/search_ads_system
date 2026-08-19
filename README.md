# Search Ads Algorithm System

一个用于模拟搜索广告/推荐广告全链路的工业级算法系统骨架。项目当前仅提供目录、模块边界、配置和接口占位；不包含模型实现，也不假设任何特定数据集（包括 Criteo）的字段定义。

## Pipeline

```text
Data Processing
      -> Recall
      -> PreRank
      -> Ranking
      -> Value Ranking
      -> Auction Simulation
      -> Evaluation
```

| 阶段 | 包 | 责任 |
| --- | --- | --- |
| Data Processing | `data` | 数据接入契约、校验、特征处理与数据集拆分 |
| Recall | `recall` | 召回接口、Two Tower 预留、FAISS ANN 检索预留 |
| PreRank | `prerank` | 轻量级候选粗排接口 |
| Ranking | `ranking` | 精排基类及 DeepFM、DCNv2、DIN 扩展位 |
| Value Ranking | `value_ranking` | 价值预估与 ESMM / 多任务 CTR-CVR 扩展位 |
| Auction Simulation | `auction` | 广告竞价与出价仿真接口 |
| Evaluation | `evaluation` | 离线效果与业务指标评估接口 |

## Layout

```text
.
├── config.yaml                    # 统一运行配置
├── requirements.txt
├── pyproject.toml
├── scripts/                       # 后续训练、评估入口
├── src/search_ads_system/
│   ├── common/                    # 配置、共享类型
│   ├── data/                      # 数据处理
│   ├── recall/                    # 召回
│   ├── prerank/                   # 粗排
│   ├── ranking/                   # 精排
│   ├── value_ranking/             # 价值排序
│   ├── auction/                   # 竞价仿真
│   ├── evaluation/                # 评估
│   └── pipelines/                 # 全链路编排
└── tests/                         # 与源码结构对应的测试目录
```

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

在接入数据时，请先在 `data/interfaces.py` 中定义本项目的数据契约，并在 `config.yaml` 填写实际路径。保持数据字段定义由数据所有者显式提供，避免依赖隐含的公开数据集 schema。

## Dataset schema check

`scripts/run_preprocess.py` 是当前的预处理入口。它按 `config.yaml` 中的显式数据契约分块扫描源文件，并输出文件大小、行数、列名、推断 dtype、缺失率和已配置标签的分布；它不会训练模型或写出特征数据。

```bash
python3.12 scripts/run_preprocess.py --config config.yaml
```

报告默认写入 `artifacts/preprocessing/schema_report.json`。Criteo Search Conversion 原始文件没有表头；配置中 23 个列名及 `Sale` 标签选择均来自随数据发布的 README，而不是由代码根据字段内容推断。

## Planned model extensions

- Recall: Two Tower、FAISS ANN Search
- Ranking: DeepFM、DCNv2、DIN
- Value Ranking: ESMM、Multi-task CTR-CVR
