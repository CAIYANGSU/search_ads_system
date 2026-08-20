# Product Ads Ranking System

面向电商商品广告的排序系统。项目以商品、用户、设备、类目和广告交互特征为输入，为广告候选的 CTR 预测、概率校准、eCPM 排序及 GSP auction simulation 提供一致的数据与模块边界。

当前已实现的是稳定、可复现的数据处理 pipeline；它不训练模型，也不执行在线排序或拍卖。现有 Criteo 转化数据只包含点击后的转化信息，适合保留商品广告的转化与价值信号。要训练可用的 CTR 模型，还需接入带曝光与点击标签的商品广告日志；该前提不会改变当前数据处理产物或实验结果。

## 排序目标与预留能力

后续 Product Ads Ranking System 将基于同一份特征产物衔接以下能力：

- **CTR prediction**：估计商品广告在给定流量与商品上下文中的点击概率；
- **probability calibration**：校准预测概率，使其与实际点击率一致，供出价与流量决策使用；
- **eCPM ranking**：以校准后的 CTR 与广告出价计算排序分数（例如 `eCPM = calibrated_pCTR × bid × 1000`）；
- **GSP auction simulation**：依据 eCPM 排序与次高价规则模拟商品广告位分配、成交价和收益。

这些是 `ranking`、`value_ranking` 和 `auction` 模块的产品定位，而非当前 pipeline 新增的执行步骤。它们不会改写或重新解释已有实验产物。

## 数据流程

```text
CriteoSearchData (TSV, raw)
        │
        ├── scripts/run_preprocess.py  → outputs/preprocessing/schema_report.json
        │
        ├── scripts/convert_criteo.py → outputs/processed/criteo_unified/part-*.csv
        │
        ├── scripts/run_eda.py        → outputs/eda/summary.json
        │                                outputs/eda/top_categories.csv
        │
        └── scripts/build_features.py → outputs/features/criteo_features/part-*.csv
                                         outputs/features/metadata.json
```

所有 pipeline 产物均由 `config.yaml` 的 `paths` 管理，并且必须位于 `paths.outputs_dir`（默认 `outputs/`）下。原始数据不会被修改。CSV 按分块写入，避免将约 6GB 的源文件一次性读入内存。

## 商品广告统一 Schema

每一行代表一次商品广告点击后的转化观测。由于原始数据没有 click ID，转换阶段基于原始行号生成稳定的 `event_id`（例如 `criteo-000000000001`）。`-1`（及 `click_timestamp` 的 `0`）转换为空值。

| 原始字段 | 统一字段 |
| --- | --- |
| `Sale` | `conversion_label` |
| `SalesAmountInEuro` | `conversion_value_eur` |
| `time_delay_for_conversion` | `conversion_delay_seconds` |
| `nb_clicks_1week` | `clicks_last_7d` |
| `click_timestamp` | `click_timestamp` |
| 商品、用户、Partner、类别字段 | 对应 `product_*`、`user_id`、`partner_id` 字段 |

完整字段定义在 `src/search_ads_system/data/unified_schema.py` 的 `UNIFIED_COLUMNS`。转换会校验 `Sale` 是二元标签，避免将异常标签带入后续商品广告排序建模步骤。

## 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认配置假设原始文件位于：

`criteo_search_conversion/Criteo_Conversion_Search/CriteoSearchData`

如文件位置不同，请只修改 `config.yaml` 中的 `preprocessing.dataset.path`；产物位置和分块大小分别由 `paths` 与 `preprocessing.dataset.chunk_size` 配置。

## 运行

请按以下顺序执行，每个命令都可独立运行并接收 `--config` 指向另一份 YAML 配置。

```bash
# 1. 仅扫描原始 schema、缺失率、推断类型与标签分布
python3.12 scripts/run_preprocess.py --config config.yaml

# 2. 生成统一 schema 的分块 CSV；首次运行无需 --overwrite
python3.12 scripts/convert_criteo.py --config config.yaml

# 3. 对统一数据生成行数、转化率、缺失率、数值统计和 Top-K 类别统计
python3.12 scripts/run_eda.py --config config.yaml

# 4. 生成基础数值、缺失指示、稳定变换、UTC 时间和类别特征
python3.12 scripts/build_features.py --config config.yaml
```

转换或特征输出目录已经存在时，命令会主动停止，防止混入不同批次产物。确认要重新生成时，显式传入 `--overwrite`：

```bash
python3.12 scripts/convert_criteo.py --config config.yaml --overwrite
python3.12 scripts/build_features.py --config config.yaml --overwrite
```

## EDA 与特征约定

EDA 使用分块累积统计，输出总体行数、转化数/转化率、每列缺失率、关键数值列的非空数/均值/最小值/最大值，及配置中低基数类别列的 Top-K 值。类别统计列由 `preprocessing.eda.categorical_columns` 控制，避免高基数 ID 造成不必要的内存占用。

特征工程不拟合全局统计量，也不使用标签构造特征，以避免数据泄漏。它保留 `conversion_label`、`conversion_value_eur` 和原始数值特征；所有数值缺失值填充为 `0`，并生成：

- `log_product_price = log1p(product_price)`、`log_clicks_last_7d = log1p(clicks_last_7d)`，以及 `conversion_delay_hours = conversion_delay_seconds / 3600`；
- `has_conversion_value`，用于标记原始 `conversion_value_eur` 是否存在；
- `click_hour_utc`、`click_day_of_week_utc` 与时间戳缺失指示；
- `device_type`、商品年龄/性别/品牌/国家、`product_category_1` 至 `product_category_4`、`audience_id`、`product_id` 与 `partner_id` 的 `cat_*` 类别特征；类别缺失值替换为 `__MISSING__`。`product_category_6` 和 `product_category_7` 因 EDA 显示缺失率超过 98% 而不再输出。

`outputs/features/metadata.json` 记录版本、特征列表、标签列和缺失类别 token，供后续商品广告 CTR/CVR 建模、概率校准、eCPM 排序和 GSP auction simulation 阶段消费。

## 定位与兼容性

项目名称已调整为 Product Ads Ranking System；Python 包名、配置键、原始数据目录和所有命令保持不变，以保证既有实验结果与运行方式完全兼容。原始目录中的 `Search` 是 Criteo 数据集的来源名称，并非项目的产品定位。

## 开发与验证

```bash
PYTHONPATH=src python3.12 -m pytest tests/data/test_criteo_pipeline.py
```

该测试用两行临时 Criteo 格式数据覆盖分块转换、EDA 与特征工程，不会读取或写入真实数据集。
