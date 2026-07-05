# DAIC-WOZ 重新分析 v2：可复现脚本包

本目录是重新分析 v2 的独立代码快照。它只包含脚本、源码、测试、冻结契约和环境信息，不重复保存原始参与者数据、模型权重或已生成的 embedding 缓存。

## 目录结构

```text
08_reproducibility_scripts/
  scripts/                 # 7 个命令行入口
  reanalysis_v2/           # 18 个实际分析模块
  tests/                   # 12 个测试文件，共 44 项测试
  frozen_contract/         # v2 方案、变更记录、输入契约、共享 split、运行清单
  documentation/           # 中英文复现指南
  requirements-lock.txt    # 核心依赖精确版本
  environment_snapshot.json
  pytest.ini               # 测试发现边界
  MANIFEST_SHA256.csv      # 本目录全部受控文件的字节数与 SHA-256
```

## 数据要求

完整重跑需要两个经过验证的输入源：

1. `mpnet_conditions_official142_verified.csv`：五个冻结主条件；
2. `turns_official142.csv`：用于重新构建并核对 C4 话轮级控制输入。

C5 控制还需要 `reviewed_c5_spans_official142.csv`。这些文件不复制进脚本包，以避免再次产生数据副本和来源混淆。其原始路径、SHA-256 和预期行数保存在 `frozen_contract/input_freeze_manifest.json` 及各运行清单中。

## 环境安装

建议使用 Python 3.12.8，并在本目录中建立独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

本次正式运行使用 `torch 2.9.1+cpu`。若安装程序不能直接解析带 `+cpu` 的版本，请按照 `requirements-lock.txt` 顶部说明，先从 PyTorch CPU wheel 索引安装 torch，再安装其余依赖。

## 首先验证代码

```powershell
python -m pytest -q
```

在仅包含脚本、不复制完整结果的独立快照中，预期结果为 `43 passed, 1 skipped`。被跳过的是依赖完整冻结结果目录的最终报告集成测试；完成全流程并生成 `analysis_v2/` 后，它会自动启用，此时预期为 `44 passed`。测试不会启动耗时的 embedding 编码。

## 完整执行顺序

下面假设输出目录为当前脚本包下的 `analysis_v2/`。请将三个输入路径替换为本机真实路径。

```powershell
python scripts/prepare_analysis_v2.py `
  --source "<path>\mpnet_conditions_official142_verified.csv" `
  --turns-source "<path>\turns_official142.csv" `
  --output analysis_v2

python scripts/run_tfidf_main.py --output analysis_v2

python scripts/run_embedding_model.py --model mpnet --device cpu --output analysis_v2
python scripts/run_embedding_model.py --model bge --device cpu --output analysis_v2
python scripts/finalize_embedding_analysis.py --output analysis_v2

python scripts/run_c5_controls.py `
  --output analysis_v2 `
  --spans "<path>\reviewed_c5_spans_official142.csv"

python scripts/run_c4_controls.py `
  --output analysis_v2 `
  --turns analysis_v2\01_inputs\c4_turns_official142_frozen.csv

python scripts/build_final_package.py --analysis-root analysis_v2
```

所有正式统计检验默认执行 10,000 次置换。调试时虽然可以通过 `--permutations` 减少次数，但此类结果不得替代正式结果。

## 冻结模型

- `sentence-transformers/all-mpnet-base-v2@e8c3b32edf5434bc2275fc9bab85f82640a19130`
- `BAAI/bge-large-en-v1.5@d4aa6901d3a41ba39fb536a557fa166f842b0e09`

脚本使用内容寻址缓存。若模型已经下载到 Hugging Face 本地缓存，可离线运行；否则首次编码需要联网下载对应 revision。

## 结果核对

- 新运行应产生自己的 `run_manifest.json` 和 `output_manifest_sha256.csv`。
- 确定性输入和 split 应与 `frozen_contract/` 中的哈希完全一致。
- 同代码、同数据、同依赖和同模型缓存下，确定性表格与 OOF 输出要求精确匹配。
- 运行时间不作为复现判定依据。
- 当前正式包的验证状态为 `ANALYZED`；由于尚未完成第二次完整的双 embedding 端到端复跑，不标记为独立重复验证成功。

## 重要解释红线

- 不把未经 FDR 校正支持的 ROC-AUC 差异写成稳定优势。
- 不把默认阈值 sensitivity 当作排序能力。
- 不把 C5 domain 状态解释为访谈者是否询问该症状域。
- 不把 C4 协议信号解释为已分离的模板自然语言语义。
- 不把官方 test 或 train+dev 写成独立外部验证。
- 两个 embedding 模型必须同时报告。

## 2026-07-05 严格来源重要性重分析

论文中的来源增量、置换重要性与临床域逐项剔除结果已改用直接原始来源联合建模。旧的受试者级聚合 OOF 概率堆叠结果仅保留作审计历史，不得再用于论文结论。

新增可复现入口：

```powershell
E:\python3.12.8\python.exe scripts\run_source_importance_strict.py
E:\python3.12.8\python.exe scripts\generate_source_importance_figures.py
```

对应代码快照为 `reanalysis_v2/source_importance_strict.py`，图形生成的规范实现为仓库根目录 `reanalysis_v2/source_importance_figures.py`。严格结果应写入 `analysis_v2/10_source_importance/07_strict_joint_models/`，图形写入 `analysis_v2/06_tables_figures/figures/`。正式运行固定为 5000 次受试者级 bootstrap、10,000 次配对置换以及三个预设检验族内的 BH-FDR 校正。
