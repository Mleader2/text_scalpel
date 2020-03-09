# coding=utf-8
# 利用文本匹配的语料，从正例中采样得到句子对(A,B),然后训练模型把Ａ改写成Ｂ
# 当前是针对LCQMC  改了几条语料的标注
import os
from compute_lcs import _compute_lcs
from curLine_file import curLine

def process(corpus_folder, raw_file_name, save_folder):
  corpus_list = []
  for name in raw_file_name:
    raw_file = os.path.join(corpus_folder, name)
    with open(raw_file, "r") as fr:
      lines = fr.readlines()

    for i ,line in enumerate(lines):
      source, target, label = line.strip().split("\t")
      if label=="0" or source==target:
        continue
      if label != "1":
        input(curLine()+line.strip())
      length = float(len(source) + len(target))
      lcs1 = _compute_lcs(source, target)
      lcs_rate= len(lcs1)/length
      if (lcs_rate<0.4):
        continue # 变动过大，忽略
      if len(source)*1.15 < len(target):
        new_t = source
        source = target
        target = new_t
        print(curLine(), source, target, ",lcs1:",lcs1 , ",lcs_rate=", lcs_rate)
      corpus = "%s\t%s\t%f\n" % (source, target, lcs_rate)
      corpus_list.append(corpus)
    print(curLine(), len(corpus_list), "from %s" % raw_file)
  save_file = os.path.join(save_folder, "lcqmc.txt")
  with open(save_file, "w") as fw:
    fw.writelines(corpus_list)
  print(curLine(), "have save %d to %s" % (len(corpus_list), save_file))

if __name__ == "__main__":
  corpus_folder = "/home/cloudminds/Mywork/corpus/Chinese_QA/LCQMC"
  raw_file_name = ["train.txt", "dev.txt", "test.txt"]
  save_folder = "/home/cloudminds/Mywork/corpus/文本复述"
  process(corpus_folder, raw_file_name, save_folder)

