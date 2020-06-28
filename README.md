# LaserTagger
一．概述  
文本复述任务是指把一句/段文本A改写成文本B，要求文本B采用与文本A略有差异的表述方式来表达与之意思相近的文本。  
改进谷歌的LaserTagger模型，使用LCQMC等中文语料训练文本复述模型，即修改一段文本并保持原有语义。  
复述的结果可用于数据增强，文本泛化，从而增加特定场景的语料规模，提高模型泛化能力。  


二．模型介绍  
谷歌在文献《Encode, Tag, Realize: High-Precision Text Editing》中采用序列标注的框架进行文本编辑，在文本拆分和自动摘要任务上取得了最佳效果。  
在同样采用BERT作为编码器的条件下，本方法相比于Seq2Seq的方法具有更高的可靠度，更快的训练和推理效率，且在语料规模较小的情况下优势更明显。  

<p align="center"><img width="50%" src="AR_architecture.png" /></p>

谷歌公开了本文献对应的代码，但是原有任务与当前任务有一定的差异性，需要修改部分代码，主要修改如下：  
A.分词方式：原代码针对英文，以空格为间隔分成若干词。现在针对中文，分成若干字。  
B.推理效率：原代码每次只对一个文本进行复述，改成每次对batch_size个文本进行复述，推理效率提高6倍。  

三.文件说明和实验步骤  
1.安装python模块  
参见"requirements.txt", "rephrase.sh"  
2.下载预训练模型  
考虑模型推理的效率，目前本项目采用RoBERTa-tiny-clue（中文版）预训练模型。  
由于目前网络上有不同版本，现将本项目使用的预训练模型上传的百度网盘。链接: https://pan.baidu.com/s/1yho8ihR9C6rBbY-IJjSagA 提取码: 2a97  
如果想采用其他预训练模型，请修改“configs/lasertagger_config.json".  
3.训练和评测模型  
根据自己情况修改脚本"rephrase.sh"中2个文件夹的路径，然后运行  bash rephrase.sh HOST_NAME
变量HOST_NAME是作者为了方便设定路径使用的，请根据自己情况修改；  
如果只是离线的对文本进行批量的泛化，可以注释脚本中其他部分，只用predict_main.py就可以满足需求。  
4.启动文本复述服务  根据自己需要，可选  
根据自己情况修改"rephrase_server.sh"文件中几个文件夹的路径，使用命令"sh rephrase_server.sh"可以启动一个文本复述的API服务.  
本API服务可以接收一个http的POST请求，解析并对其中的文本进行泛化，具体接口请看“rephrase_server/rephrase_server_flask.py"  

文本复述的语料需要自己整理语义一致的文本对。如果用自己业务场景下的语料最好，当然数量不能太少，如果没有或不够就加上LCQMC等语料中的正例。  
然后用最长公共子串的长度限制一下，因为这个方法要求source和target的字面表达不能差异过大，可以参考一下“get_text_pair_lcqmc.py”。  
目前，我的train.txt,tune.txt中都是三列即text1,text2,lcs_score,之间用tab"\t"分割。  

有几个脚本文件如rephrase_for_qa.sh，rephrase_for_chat.sh，rephrase_for_skill.sh是作者自己办公需要的，可以忽略  

四．实验效果  
1. 在公开数据集Wiki Split上复现模型：  
Wiki Split数据集是英文语料，训练模型将一句话拆分成两句话，并保持语义一致，语法合理，语义连贯通顺，如下图所示。  

<p align="center"><img width="50%" src="sentence_fusion_task.png" /></p>

Exact　score=15,SARI score=61.5,KEEP score=93,ADDITION score=32,DELETION score=59,  
基本与论文中的Exact score=15.2；SARI score=61.7一致（这些分数均为越高越好）。  
2. 在自己构造的中文数据集训练文本复述模型：  
（1）语料来源  
（A）一部分语料来自于LCQMC语料中的正例，即语义接近的一对文本；  
（B）另一部分语料来自于业务ＦＡＱ下面同一答案的问题;   
因为模型的原理，要求文本A和B在具有一定的重合字数，故过滤掉上述两个来源中字面表述差异大的文本，如“我要去厕所”与“卫生间在哪里”。对语料筛选后对模型进行训练和测试。  
（2）测试结果：  
对25918对文本进行复述和自动化评估，评测分数如下（越高越好）：  
Exact score=29,SARI score=64,KEEP score=84,ADDITION score=39,DELETION score=66.  
CPU上耗时0.5小时，平均复述一句话需要0.72秒。  
可能是语言和任务不同，在中文文本复述上的评测分数比公开数据集高一些。  

五.一些trick  
1.可以设定对于某些字或词不做修改  
如对实体识别NER的语料泛化，需要保证模型不能修改其中的实体；  
对业务语料泛化，也可以根据情况保证模型不能修改其中的关键字 如日期，航班号等；  
目前，是通过正则的方式定位这些不能被模型修改的位置，然后将这些位置的location设置为1，具体实现参见tagging.py.  
2.增加复述文本与原文本的差异度  
可以对训练语料中的text_a先进行随机的swag操作，相应地脚本中enable_swap_tag改为true，再训练模型将其改写为text_b;  
实际应用或测试时同样将原始文本text_a先进行随机的swag操作，然后利用模型改写为text_b;  
因为训练语料中text_a是不通顺，但text_b是通顺的，所以实际应用或测试时仍然会得到通顺的复述结果。  

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
