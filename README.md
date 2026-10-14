# A 股组合风控与风险预测

组合风险度量、限额检查、压力测试与机器学习扩展。

## 内容

- `run_workflow.py`、`src/`、`config/`：日线数据、组合构建、VaR/CVaR、EWMA、回撤、Beta、限额和压力测试。
- `ml_extension/`：聚宽面板数据、特征、时间顺序训练及模型评估脚本。

## 运行

```bash
python -m pip install -r requirements.txt
python run_workflow.py
```

Python 3.10+。运行时抓取行情；接口失败时原工作流会生成标注为虚拟的数据，必须查看输出中的数据来源。回测起点为 2020-01-23。输出在 `output/`。

机器学习扩展的依赖在 `ml_extension/requirements.txt`，聚宽环境脚本需要对应账户权限；本仓库没有凭据或已经验证的样本外收益声明。

## 项目说明

维护者：[lhowd301-lab](https://github.com/lhowd301-lab)。本仓库整理自既有课程、研究或工具项目，开发与文档整理使用了 AI 辅助。代码可运行性与研究结论有效性分别说明，不将历史模拟结果视为实盘业绩。
