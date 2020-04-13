# LaserTagger


一．概述
文本复述任务是指把一句/段文本A改写成文本B，要求文本B采用与文本A略有差异的表述方式来表达与之意思相近的文本。
#改进谷歌的LaserTagger模型，使用LCQMC等中文语料训练文本复述模型，即修改一段文本并保持原有语义。
复述的结果可用于数据增强，文本泛化，从而增加特定场景的语料规模，提高模型泛化能力。
#根据自己情况，修改"rephrase_server.sh"文件中的HOST_NAME等参数，使用命令"sh rephrase_server.sh"可以启动一个文本复述的服务
#有几个脚本文件如rephrase_for_qa.sh，rephrase_for_chat.sh，rephrase_for_skill.sh是作者自己办公需要的，可以忽略


二．模型介绍

谷歌在文献《Encode, Tag, Realize: High-Precision Text Editing》中采用序列标注的框架进行文本编辑，在文本拆分和自动摘要任务上取得了最佳效果。
在同样采用BERT作为编码器的条件下，本方法相比于Seq2Seq的方法具有更高的可靠度，更快的训练和推理效率，且在语料规模较小的情况下优势更明显。

<p align="center"><img width="50%" src="AR_architecture.png" /></p>

谷歌公开了本文献对应的代码，但是原有任务与当前任务有一定的差异性，需要修改部分代码，主要修改如下：
A.分词方式：原代码针对英文，以空格为间隔分成若干词。现在针对中文，分成若干字。
B.推理效率：原代码每次只对一个文本进行复述，改成每次对batch_size个文本进行复述，推理效率提高6倍。



三．实验效果
1. 在公开数据集Wiki Split上复现模型：
Wiki Split数据集是英文语料，训练模型将一句话拆分成两句话，并保持语义一致，语法合理，语义连贯通顺，如下图所示。

<p align="center"><img width="50%" src="sentence_fusion_task.png" /></p>

Exact　score=15,SARI score=61.5,KEEP score=93,ADDITION score=32,DELETION score=59,
基本与论文中的Exact score=15.2；SARI score=61.7一致（这些分数均为越高越好）。
2. 在自己构造的中文数据集训练文本复述模型：
（1）语料来源
（A）一部分语料来自于LCQMC语料中的正例，即语义接近的一对文本；
（B）另一部分语料来自于宝安机场用户QA下面同一答案的问题。
因为模型的原理，要求文本A和B在具有一定的重合字数，故过滤掉上述两个来源中字面表述差异大的文本，如“我要去厕所”与“卫生间在哪里”。筛选后
（2）测试结果：
对25918对文本进行复述和自动化评估，评测分数如下（越高越好）：
Exact score=29,SARI score=64,KEEP score=84,ADDITION score=39,DELETION score=66.
CPU上耗时0.5小时，平均复述一句话需要0.72秒。
可能是语言和任务不同，在中文文本复述上的评测分数比公开数据集高一些。

## How to Cite LaserTagger

```
@inproceedings{malmi2019lasertagger,
  title={Encode, Tag, Realize: High-Precision Text Editing},
  author={Eric Malmi and Sebastian Krause and Sascha Rothe and Daniil Mirylenka and Aliaksei Severyn},
  booktitle={EMNLP-IJCNLP},
  year={2019}
}
```

## License

Apache 2.0; see [LICENSE](LICENSE) for details.
