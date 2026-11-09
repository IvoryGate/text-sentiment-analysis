# text-sentiment-analysis

使用 IMDB 英文影评比较传统机器学习、CNN/LSTM、DistilBERT 和模型投票。各方案的验证集、测试集指标与推理成本见 [Notebook 026](notebooks/026_project_experiment_summary.ipynb) 和[项目实验总表](reports/tables/project_experiment_summary.csv)。

## 复现 Notebook 023

从包含本项目实验结果的检出开始，在仓库根目录使用 Python 3.12 安装依赖和项目。

```shell
python -m pip install -r requirements.txt
python -m pip install -e . joblib==1.4.2
```

准备[固定测试协议](configs/three_model_voting_test_protocol.json)中列出的原始文件。脚本按协议核对文件的 SHA-256，因此请保留归档文件的原始字节。

| 输入 | 来源 |
| --- | --- |
| `data/cleaned/test.csv` | Notebook 001–002 的清洗后测试集 |
| `reports/tables/fixed_models_test_predictions.csv` | Notebook 015 导出的 B06 和 TextCNN 测试预测 |
| `reports/tables/distilbert_test_predictions.csv` | Notebook 020 导出的 DistilBERT 测试预测 |
| `data/models/b06_sigmoid_cv5.joblib` | Notebook 022 导出的 B06 sigmoid 校准模型 |
| `configs/final_test_protocol.json`、`configs/distilbert_test_protocol.json` | Notebook 015、020 对应的历史测试协议 |

`data/` 由 Git 忽略。把原工作区的 `data/archives/notebook_023_inputs.zip` 复制到新检出根目录之外，再从新检出的根目录解包。归档包含表中的两份 `data/` 文件，路径已包含 `data/` 前缀。

```shell
python -m zipfile -e ../notebook_023_inputs.zip .
```

从仍有原始输入的 Windows 工作区重新制作归档时，在仓库根目录执行以下两条命令。

```shell
python -c "from pathlib import Path; Path('data/archives').mkdir(parents=True, exist_ok=True)"
tar.exe -a -cf data/archives/notebook_023_inputs.zip data/cleaned/test.csv data/models/b06_sigmoid_cv5.joblib
```

如果需要重建实验输入，可沿 Notebook 001–016、019–020、021–022 的顺序生成。Notebook 019–020 的基础模型版本、CUDA BF16 和软件版本要求记录在 [DistilBERT 训练配置](configs/distilbert_finetuning.json)及对应 Notebook 中。不同环境重建的模型或 CSV 可能有不同的文件指纹，严格复现 023 使用归档输入。

```shell
python scripts/evaluate_three_model_voting.py
```

脚本与 [Notebook 023](notebooks/023_three_model_voting_test_evaluation.ipynb) 共用评估逻辑。它在 CPU 上计算 B06 的校准概率，读取已保存的 TextCNN 和 DistilBERT 预测，对齐 24,678 条测试评论，并计算硬投票、等权软投票和固定加权软投票。固定权重按 B06、TextCNN、DistilBERT 的顺序为 `0.2 / 0.3 / 0.5`，正类概率严格大于 `0.5` 时判为正类。

默认把指标和逐条预测保存到 `reports/reproduced/`，也可以用 `--output-dir` 指定目录。三种投票的预期测试 F1 依次为 **91.45%**、**92.07%**、**92.25%**。Git 对固定协议、输入预测和参考 CSV 保留原始字节，脚本按 Notebook 023 的 CRLF 格式写出结果。

用下面的命令逐字节比较两份复现结果与 `reports/tables/` 下的同名参考文件。

```shell
python -c "from pathlib import Path; names = ('three_model_voting_test_comparison.csv', 'three_model_voting_test_predictions.csv'); assert all(Path('reports/reproduced', name).read_bytes() == Path('reports/tables', name).read_bytes() for name in names); print('Both CSVs match byte for byte.')"
```
