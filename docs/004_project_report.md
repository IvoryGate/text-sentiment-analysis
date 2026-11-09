# IMDB 英文影评情感分类项目报告

## 摘要

本项目在同一份清洗后的 IMDB 数据上比较 TF-IDF 线性分类器、TextCNN、LSTM、DistilBERT 及其融合方案。B06（TF-IDF + 线性 SVM）的测试集 F1 为 89.01%，高于 TextCNN、单向 LSTM 和双向 LSTM。DistilBERT 是本项目测试集 F1 最高的单模型，得分为 90.90%。融合 B06、TextCNN 和 DistilBERT 后，硬投票、等权软投票和固定加权软投票的测试集 F1 分别为 91.45%、92.07% 和 92.25%。加权相对等权的测试集 F1 差值点估计为 +0.18 个百分点，95% 配对 bootstrap 区间为 −0.012 至 +0.365 个百分点，跨过零。固定加权软投票比 DistilBERT 单模型高 1.34 个百分点，在本机实测中单条中位延迟从 7.32 ms 增至 9.79 ms。 [项目实验总表](../reports/tables/project_experiment_summary.csv) · [推理成本对照](../reports/tables/inference_benchmark_comparison.csv)

## 1. 引言

影评情感分类要求根据整条评论判断其总体倾向。评论中的情感信号既可能出现在短语中，也可能通过否定、转折或前后文表达。本项目用同一数据划分比较稀疏词项特征、卷积序列模型、循环序列模型和预训练语言模型，考察它们的分类效果和错误差异。

实验围绕三个问题展开。第一，TextCNN、LSTM 和 DistilBERT 与传统基线相比表现如何。第二，模型预测的差异能否转化为融合收益，硬投票、等权软投票和加权软投票分别带来多少改善。第三，融合获得的性能提升需要付出多少推理成本。 [原项目计划](00_project_plan.md)

## 2. 相关工作

### 2.1 影评情感分类与线性基线

Pang 等人较早将朴素贝叶斯、最大熵模型和支持向量机用于影评分类，并指出情感分类与主题分类的难点不同［1］。Maas 等人随后发布 IMDB 大规模影评数据，并研究了词向量在情感分类中的作用［2］。Wang 和 Manning 在多个数据集上比较了线性分类器和词项特征，发现加入二元词项（bigram）可以改善其情感分类基线［3］。本文比较一元词项（unigram）、二元词项及二者的组合，考察只使用相邻词对和在一元词项中加入相邻词对这两种做法。

### 2.2 序列模型与预训练语言模型

Kim 将卷积网络用于句子分类，用局部卷积提取短语特征，再通过最大池化汇总各位置的响应［4］。Hochreiter 和 Schmidhuber 提出的 LSTM 用门控状态传递序列信息［5］。本项目在影评级别比较这两类模型，并用单向和双向 LSTM 对照不同方向的上下文建模。

BERT 先在通用语料上进行双向 Transformer 预训练，再针对下游任务微调［6］。DistilBERT 在预训练阶段通过知识蒸馏得到更紧凑的模型［7］。本项目从现有 DistilBERT 检查点出发微调，将其同时作为单模型对照和投票成员。这个设置比较的是完整训练和输入方案，预训练本身也是模型差异的一部分。

### 2.3 分类器融合与概率校准

Kittler 等人讨论了分类器的多数票和分数求和等组合规则。硬投票使用各模型给出的类别，软投票则组合各模型的正类概率［8］。线性 SVM 提供决策分数，因此本文使用 scikit-learn 实现的 Platt sigmoid 校准将其映射为概率，再与神经网络的 softmax 输出组合［9–10］。等权软投票与固定加权软投票共用三组概率、模型检查点和阈值，可以在同一批样本上比较权重调整带来的变化。

## 3. 数据集与实验方法

### 3.1 数据清洗与划分

对原始训练集和测试集分别按评论文本去重，并从测试集中移除与训练集重复的评论。使用随机种子 42 按标签分层，从去重后的训练数据划出 10% 作为验证集。随后清理部分 HTML 标签。最终训练集、验证集和测试集分别有 22,413、2,491 和 24,678 条评论。去重后的测试集与原始 IMDB 测试集规模不同。 [数据清洗与划分脚本](../scripts/clean_and_split_data.py) · [Notebook 002](../notebooks/002_data_cleaning.ipynb)

**表 1　清洗后的 IMDB 数据划分（条）**

| 划分 | 负面评论 | 正面评论 | 合计 |
| --- | ---: | ---: | ---: |
| 训练集 | 11,189 | 11,224 | 22,413 |
| 验证集 | 1,243 | 1,248 | 2,491 |
| 测试集 | 12,266 | 12,412 | 24,678 |

### 3.2 输入表示与模型训练

TF-IDF 只在训练集上拟合，再用于转换验证集和测试集。一元词项、二元词项和组合特征都使用 `min_df=2` 和 `max_features=20000`。组合特征共享 20,000 个位置。三种表示各自搭配逻辑回归和线性 SVM，共得到六组基线。六组使用 `C=1.0` 和 `max_iter=1000`，另在组合特征上线性 SVM 比较 `C=0.1、1.0、10.0`。 [传统基线设置](003_traditional_machine_learning.md) · [Notebook 004](../notebooks/004_traditional_ml.ipynb)

TextCNN 和 LSTM 共用训练集词频最高的 20,000 个词项。预处理保留每条评论前 512 个词项，词表外词项映射为 `<UNK>`，较短序列在末尾补 `<PAD>`。模型都使用 128 维词嵌入。固定 TextCNN 使用 64 个宽度为 3 的卷积核和最大池化。LSTM 的隐状态维度为 128，单向模型取最后一个有效状态，双向模型拼接两个方向的最终状态。两类模型采用交叉熵和 Adam，学习率为 0.001、批大小为 32，最多训练 20 轮。验证损失连续四轮未改善时早停，测试采用最低验证损失对应的检查点。 [序列准备](../notebooks/005_sequence_preparation.ipynb) · [TextCNN 训练](../notebooks/006_textcnn.ipynb) · [LSTM 结构对照](../notebooks/012_lstm_architecture_experiments.ipynb)

DistilBERT 使用对应检查点的子词分词器，输入最多 256 个词项，含特殊词项。微调更新全部模型参数，使用 AdamW，学习率为 0.00002、权重衰减为 0.01。每批 16 条，累积两步梯度，以 BF16 训练两轮并按最低验证交叉熵选检查点。 [DistilBERT 训练配置](../configs/distilbert_finetuning.json) · [Notebook 019](../notebooks/019_distilbert_finetuning.ipynb)

### 3.3 概率校准、权重选择与评价

B06 的概率校准在训练集内完成。先做五折分层划分，每折仅用其余四折重新拟合 TF-IDF 和线性 SVM，再为留出折生成决策分数。用汇总的折外分数和对应标签拟合一个 sigmoid。实际预测时由全训练集重新拟合的 TF-IDF 和 SVM 提供分数，再通过该 sigmoid 得到正类概率。TextCNN 和 DistilBERT 使用各自检查点的 softmax 正类概率。硬投票使用三模型原有的类别预测。 [Notebook 022](../notebooks/022_three_model_soft_voting.ipynb) · [Notebook 023](../notebooks/023_three_model_voting_test_evaluation.ipynb)

等权软投票的三个权重各为 1/3。加权搜索要求每个权重至少为 0.1，按 0.1 步长枚举总和为 1 的组合，同时纳入等权方案。按验证集 F1 选择权重，并列时优先选择更接近等权的组合。另对验证集做五折选权和留出预测，估计选权流程在留出评论上的表现。训练集五折用于拟合校准器，验证集五折用于评估选权流程，两者使用的数据和目的不同。 [Notebook 022](../notebooks/022_three_model_soft_voting.ipynb) · [各折所选权重](../reports/tables/three_model_soft_voting_fold_weights.csv)

以正面评论（标签 1）为正类。准确率（Accuracy）统计总体判对比例，精确率（Precision）统计预测为正面的样本中判对的比例，召回率（Recall）统计实际正面样本中被识别的比例。主指标 F1 是精确率与召回率的调和平均。各指标以百分比表示，差值根据未舍入的数值计算，以百分点表示。每轮测试按对应的固定协议执行，三模型权重、阈值及成员检查点均在该轮测试前确定。比较两种软投票时，以同一条评论为配对单位，统计预测变化和 F1 差值的配对重采样区间。 [早期固定测试协议](../configs/final_test_protocol.json) · [三模型固定测试协议](../configs/three_model_voting_test_protocol.json)

## 4. 基线实验

传统六组基线中，组合词项特征搭配线性 SVM 的 B06 验证集 F1 为 88.89%，是本项目选用的传统对照。仅使用一元词项的线性 SVM 为 88.85%，两者相差约 0.04 个百分点。单独使用二元词项的两个配置分别低于其对应的一元词项配置。 [传统基线验证结果](../reports/tables/traditional_ml_validation_metrics.csv)

**表 2　基线模型的分类指标**

| 模型 | 验证集 F1 | 测试集准确率 | 测试集 F1 |
| --- | ---: | ---: | ---: |
| B06（TF-IDF + 线性 SVM） | 88.89% | 88.95% | 89.01% |
| TextCNN | 86.86% | 85.98% | 85.91% |
| 单向 LSTM | 84.81% | 84.73% | 84.03% |
| 双向 LSTM | 85.09% | 84.95% | 85.16% |

在本项目选定的配置下，B06 的测试集 F1 比 TextCNN 高 3.09 个百分点。双向 LSTM 比单向 LSTM 高 1.13 个百分点。TextCNN 的随机失活（Dropout）消融使用两组配对种子，启用随机失活后，验证集 F1 分别高 0.23 和 0.57 个百分点。这组结果对应以最低验证损失选择检查点的规则。 [项目实验总表](../reports/tables/project_experiment_summary.csv) · [TextCNN 随机失活配对结果](../reports/tables/textcnn_dropout_ablation_paired.csv)

## 5. 集成学习分析

### 5.1 TextCNN 与 LSTM 的互补性

TextCNN 与单向或双向 LSTM 的等权软投票，测试集 F1 分别为 87.04% 和 87.84%。两组融合都超过各自成员，但仍低于 B06 的 89.01%。以种子 42 的 TextCNN 为验证集参照，单向融合纠正 117 条并新增 75 条错误，双向融合也纠正 117 条，但新增 95 条错误。两组融合分别净少错 42 条和 22 条。 [方向融合错误对照](../reports/tables/cnn_lstm_direction_error_comparison.csv) · [项目实验总表](../reports/tables/project_experiment_summary.csv)

### 5.2 三模型投票

三模型方案使用 B06、TextCNN 和 DistilBERT。硬投票在至少两票为正类时判为正类。软投票的加权平均得分严格大于 0.5 时判为正类。完整验证集选出的权重按 B06、TextCNN、DistilBERT 的顺序为 0.2、0.3、0.5。 [Notebook 022](../notebooks/022_three_model_soft_voting.ipynb)

**表 3　三模型投票的分类指标**

| 三模型方案 | 验证集 F1 | 测试集准确率 | 测试集 F1 | 相对硬投票的测试集 F1 差值 |
| --- | ---: | ---: | ---: | ---: |
| 硬投票 | 91.79% | 91.37% | 91.45% | 0.00 个百分点 |
| 等权软投票 | 92.52% | 92.01% | 92.07% | +0.62 个百分点 |
| 固定加权软投票 | 92.87% | 92.15% | 92.25% | +0.79 个百分点 |

固定加权软投票的 92.87% 来自同时用于选权和评分的完整验证集。将验证集再分为五折，每次在其余四折选权、在留出折预测，合并后得到 92.71% 的折外 F1。固定权重则在测试集上取得 92.25%。 [验证集投票结果](../reports/tables/three_model_soft_voting_validation.csv) · [固定投票测试结果](../reports/tables/three_model_voting_test_comparison.csv)

固定加权软投票相对硬投票的测试集 F1 差值为 +0.79 个百分点。其中，从硬投票改为等权软投票的差值为 +0.62 个百分点，进一步调整权重的差值为 +0.18 个百分点。以硬投票为参照，等权软投票纠正 333 条并新增 176 条错误，固定加权软投票纠正 504 条并新增 311 条错误，分别净少错 157 条和 193 条。 [投票错误变化](../reports/tables/three_model_voting_test_comparison.csv)

### 5.3 成员消融

每次移除一个模型，将剩余权重按原比例归一化，在同一验证集上重新比较。去掉 DistilBERT 时降幅最大。去掉 B06 得到的 TextCNN + DistilBERT 是三组双模型中得分最高的一组，比硬投票高 0.38 个百分点，保留约 35% 的完整融合增益。去掉 TextCNN 后，B06 + DistilBERT 比三模型硬投票低 0.20 个百分点。 [成员消融结果](../reports/tables/three_model_voting_ablation_validation.csv)

**表 4　三模型投票的验证集成员消融**

| 方案 | 验证集 F1 | 相对完整融合的验证集 F1 差值 |
| --- | ---: | ---: |
| 固定加权软投票 | 92.87% | 0.00 个百分点 |
| 去掉 B06 | 92.17% | −0.69 个百分点 |
| 去掉 TextCNN | 91.59% | −1.27 个百分点 |
| 去掉 DistilBERT | 89.20% | −3.67 个百分点 |

## 6. DistilBERT 扩展实验

以英文 DistilBERT 预训练权重初始化模型，加入二分类头，用种子 42 微调两轮。第 1 轮的验证集交叉熵为 0.2290，低于第 2 轮的 0.2364，因此按预先设定的规则选用第 1 轮检查点。该模型验证集 F1 为 90.88%，测试集 F1 为 90.90%，比 B06 的测试集 F1 高 1.90 个百分点。在验证集中，输入上限为 256 时有 1,027 条评论发生截断，占 41.23%。 [微调历史](../reports/tables/distilbert_training_history.csv) · [输入长度统计](../reports/tables/distilbert_token_coverage.csv) · [Notebook 020](../notebooks/020_distilbert_test_evaluation.ipynb)

## 7. 讨论

### 7.1 分类效果与推理成本

计时使用同一批 256 条验证评论，每个方案测四轮，包含分词、特征转换、设备传输、前向推理和投票，模型加载、读取数据和预热在计时前完成。B06 运行在 Intel Core i9-13900HX CPU 上，TextCNN 与 DistilBERT 运行在 RTX 4060 Laptop GPU 上，融合依次执行各模型。单条延迟使用批大小 1，吞吐量使用批大小 16。P95 是单条请求耗时的第 95 百分位数。 [Notebook 025](../notebooks/025_inference_benchmark.ipynb) · [计时环境](../reports/tables/inference_benchmark_metadata.json)

**表 5　推理成本与测试集 F1**

| 方案 | 单条中位延迟 | 单条 P95 延迟 | 批量吞吐量（条/s） | 测试集 F1 |
| --- | ---: | ---: | ---: | ---: |
| B06 | 0.59 ms | 1.22 ms | 3,165 | 89.01% |
| DistilBERT | 7.32 ms | 10.93 ms | 582 | 90.90% |
| TextCNN + DistilBERT | 8.36 ms | 12.49 ms | 539 | — |
| 三模型固定加权软投票 | 9.79 ms | 14.75 ms | 432 | 92.25% |

相对 DistilBERT，三模型固定加权软投票的测试集 F1 高 1.34 个百分点，单条中位延迟增加约 34%，批量吞吐量下降约 26%。TextCNN + DistilBERT 的推理成本介于两者之间，其验证集 F1 为 92.17%，表中的破折号表示该方案仅在验证集评估。 [推理成本对照](../reports/tables/inference_benchmark_comparison.csv)

### 7.2 加权相对等权的增益

在 24,678 条测试评论上，两种软投票有 574 条预测不同，占 2.33%。以等权软投票为参照，固定加权软投票纠正了 305 条，同时新增了 269 条错误，净少错 36 条。固定加权软投票多识别出 69 条真实正面评论，也多将 33 条负面评论判为正面。正类召回率从 92.23% 升至 92.78%，精确率从 91.91% 降至 91.72%。因此，这 0.18 个 F1 百分点对应当前阈值下召回率上升、精确率小幅下降的权衡。 [测试指标](../reports/tables/three_model_voting_test_comparison.csv) · [逐条预测](../reports/tables/three_model_voting_test_predictions.csv)

为估计差值的抽样波动，以评论为单位做配对重采样［11］。每次有放回抽取 24,678 条样本，保留同一评论的真实标签和两种预测，在同一份重采样上计算两个 F1 后取差。固定随机种子 42，重复 50,000 次，取差值分布的第 2.5 和第 97.5 百分位，得到加权减等权的 95% 配对 bootstrap 区间 **−0.012 至 +0.365 个百分点**。区间跨过零，**+0.18 个百分点是本次测试的点估计**，换一批同类评论后加权能否稳定占优仍不确定。该区间反映固定模型和权重下的评论抽样波动，训练种子、重新选权和跨域变化需要另行评估。 [测试逐条预测](../reports/tables/three_model_voting_test_predictions.csv)

### 7.3 局限性与复现

**数据与模型。** 本文使用去重后的单一英文影评数据集，结论适用于当前划分和标签定义。不同模型还使用了不同的预训练、分词、输入长度和训练配置，性能差异难以单独归因于模型结构。DistilBERT 只使用一个固定种子微调，约四成验证评论在 256 词项处截断。输入长度以及被截去的评论尾部可能影响模型之间的比较。 [输入长度统计](../reports/tables/distilbert_token_coverage.csv) · [Notebook 026](../notebooks/026_project_experiment_summary.ipynb)

**选权与消融。** 92.87% 来自同时用于选权和评分的完整验证集。92.71% 的折外 F1 针对每折重新选权的流程，成员消融也使用这批已参与选择的验证样本。各阶段测试结果陆续查看，最终三模型的权重在该轮测试前固定。逐成员贡献对应当前权重归一化和固定阈值，其他权重、模型组合或部署数据可能改变结果。 [Notebook 022](../notebooks/022_three_model_soft_voting.ipynb) · [Notebook 024](../notebooks/024_three_model_voting_ablation.ipynb)

**推理成本。** 计时反映单机、固定批大小和顺序执行的稳态请求，模型加载和并发排队会改变部署中的延迟。等权和加权使用相同的三个模型，开销主要来自获取这三组预测。第 7.1 节的 34% 延迟增幅比较的是完整融合与 DistilBERT 单模型。 [Notebook 025](../notebooks/025_inference_benchmark.ipynb)

固定投票评估可以在干净 Git 检出中用归档输入复现。评估脚本按协议校验数据、校准模型和已保存预测的哈希，在 CPU 上重新计算 B06 的校准概率，复用两种神经模型的预测。独立检出的汇总指标和 24,678 条逐样本预测均与 Notebook 023 的参考 CSV 逐字节一致。 [复现步骤](../README.md) · [评估脚本](../scripts/evaluate_three_model_voting.py)

## 8. 结论

在本项目的 IMDB 划分上，B06 的测试集 F1 高于 TextCNN 和 LSTM，且在本机计时中单条中位延迟低于 1 ms。CNN/LSTM 融合改善了各自成员的结果。B06、TextCNN 和 DistilBERT 的三种投票方案在测试集上均超过了本项目得分最高的单模型 DistilBERT。从硬投票改为等权软投票，占硬投票到固定加权软投票这段测试集 F1 增幅的大部分。

固定加权软投票取得本次比较中最高的测试集 F1（92.25%）。它相对等权软投票的差值为 +0.18 个百分点，95% 配对 bootstrap 区间跨过零，现有结果不足以判断这一优势能否稳定复现。相对 DistilBERT 单模型的 1.34 个百分点提升，则需要与本机实测中约 34% 的单条中位延迟增幅一起衡量。

## 参考文献

1. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment Classification using Machine Learning Techniques. EMNLP, 79–86. [原文](https://aclanthology.org/W02-1011/)
2. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. ACL-HLT, 142–150. [原文](https://aclanthology.org/P11-1015/)
3. Wang, S., & Manning, C. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL, 90–94. [原文](https://aclanthology.org/P12-2018/)
4. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP, 1746–1751. [原文](https://aclanthology.org/D14-1181/)
5. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780. [原文](https://pubmed.ncbi.nlm.nih.gov/9377276/)
6. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, 4171–4186. [原文](https://aclanthology.org/N19-1423/)
7. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108. [原文](https://arxiv.org/abs/1910.01108)
8. Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On Combining Classifiers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(3), 226–239. [原文](https://ieeexplore.ieee.org/abstract/document/667881/)
9. Platt, J. C. (2000). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. In Advances in Large-Margin Classifiers, 61–74. MIT Press. [作者预印本](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf)
10. scikit-learn developers. Probability calibration. scikit-learn 1.8 文档。 [文档](https://scikit-learn.org/1.8/modules/calibration.html)
11. Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing. ACL, 1383–1392. [原文](https://aclanthology.org/P18-1128/)

## 附录 A　配对重采样复现

在仓库根目录运行以下 Python 代码。先按真实标签、等权预测和加权预测统计八种联合结果的频数，再从对应的多项分布抽样。对这里的二分类 F1，这与逐条有放回抽取评论得到相同的重采样分布。

```python
import numpy as np
import pandas as pd

columns = ["LABEL", "Equal soft vote", "Weighted soft vote"]
path = "reports/tables/three_model_voting_test_predictions.csv"
predictions = pd.read_csv(path, usecols=columns)[columns].to_numpy()
groups, counts = np.unique(predictions, axis=0, return_counts=True)
labels, equal, weighted = groups.T
rng = np.random.default_rng(42)
samples = rng.multinomial(len(predictions), counts / len(predictions), size=50_000)


def f1(frequencies, predicted):
    true_positives = frequencies @ (labels * predicted)
    return 2 * true_positives / (frequencies @ (labels + predicted))


point = 100 * (f1(counts, weighted) - f1(counts, equal))
differences = 100 * (f1(samples, weighted) - f1(samples, equal))
print(point, np.percentile(differences, [2.5, 97.5]))
```
