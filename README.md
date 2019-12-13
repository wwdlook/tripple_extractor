# tripple_extractor
Modified from projet：https://github.com/liuhuanyong/EventTriplesExtraction

为了分析论坛平台或者电商平台上用户的评论文本中所涉及的评论实体以及相关的情感判断做的小demo，是基于liuhuanyong的EventTriplesExtraction项目。  
初步思路是借助LTP工具做语义分析进行情感标签的机器自动标注，进而使用标注好的样本进行下一步的模型训练迭代。最终完成一个保证一定效果的可用的冷启动情况下的网络评论情感标签分类流程。  
本个demo目前对liuhuanyong的EventTriplesExtraction项目进行了一些改进。改进的点主要针对网络评论缺失主语的情况。 
接下来会持续补充自动收集辅助分析评论主体及其情感标签脚本，以及模型设置&训练脚本。

# 脚本功能介绍
### 1 sentence_parser
借助pyltp工具进行语义分析，合并存在RAD关系的切词（主要是为了后续三元组抽取时，抽出的项的语义完整性）

### 2 trple_extraction
利用语义分析后的成果，根据语义规则抽取评论三元组

To Be Continued...