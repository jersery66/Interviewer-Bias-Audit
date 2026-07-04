# 测试说明

源码测试与正式工作区保持一致。脚本快照不复制完整计算结果，因此 `conftest.py` 只在 `analysis_v2/02_tfidf_main/metrics/primary_metrics.csv` 不存在时，跳过依赖全套结果的 `test_build_final_package_from_frozen_outputs`。

- 仅脚本快照：预期 `43 passed, 1 skipped`。
- 完成全流程并生成结果后：预期 `44 passed`。

除这一项有明确条件的集成测试外，任何失败均应视为代码或环境不一致。
