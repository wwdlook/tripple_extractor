#!/usr/bin/env python3
# coding: utf-8
# File: sentence_parser.py


import os
import sys
from collections import defaultdict
from pyltp import Segmentor, Postagger, Parser, SementicRoleLabeller
_current_directory = os.path.join(os.getcwd(), 'helpers')
sys.path.append(_current_directory)

LTP_MODEL_DIR = os.path.abspath('')
# LTP_DICT_DIR = os.path.join(LTP_MODEL_DIR, 'cws_user.dict')
# LTP_POS_DICT_DIR = os.path.join(LTP_MODEL_DIR, 'pos_user.dict')

seg_model_path = os.path.join(LTP_MODEL_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_MODEL_DIR, 'pos.model')
par_model_path = os.path.join(LTP_MODEL_DIR, 'parser.model')
pisrl_model_path = os.path.join(LTP_MODEL_DIR, 'pisrl_win.model')


class LtpParser:
    def __init__(self):
        self.segmentor = Segmentor()
        # self.segmentor.load_with_lexicon(seg_model_path, LTP_DICT_DIR)
        self.segmentor.load(seg_model_path)

        self.postagger = Postagger()
        # self.postagger.load_with_lexicon(pos_model_path, LTP_POS_DICT_DIR)
        self.postagger.load(pos_model_path)

        self.parser = Parser()
        self.parser.load(par_model_path)

        self.labeller = SementicRoleLabeller()
        self.labeller.load(pisrl_model_path)

    # 语义角色标注
    def format_labelrole(self, words, postags):
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            roles_dict[role.index] = {arg.name:[arg.name,arg.range.start, arg.range.end] for arg in role.arguments}
            # roles_dict[role.index] = defaultdict(list)
            # for arg in role.arguments:
            #     roles_dict[role.index][arg.name].append((arg.name,arg.range.start, arg.range.end))
        return roles_dict

    # 句法分析, 为每个切词维护一个保存句法依存子节点的字典
    def build_parse_child_dict(self, words, postags, arcs):
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index+1:   #arcs的索引从1开始
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list

    # parser主函数
    def parser_main(self, sentence, need_seg=True):
        """
        :param sentence:
        :param need_seg: default is True, False means that sentence has been cut into words before being input for sure
        :return:
        """
        pos_dict = {
            'boost': 'n',
            'Foam': 'n'
        }
        # if isinstance(sentence, unicode):
        #     sentence = sentence.encode('utf-8')
        if need_seg or type(sentence) is str:
            words = [w for w in self.segmentor.segment(sentence)]
        else:
            words = sentence[:]
        # words = ['打', '完', '一', '场', '实战', '觉得', '鞋子', '的', '护', '脚踝', '这', '方面', '不行']
        postags = [pos for pos in self.postagger.postag(words)]
        postags = [pos_dict.get(words[index], x) for index, x in enumerate(postags)]
        arcs = self.parser.parse(words, postags)
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(words, postags)
        return words, postags, child_dict_list, roles_dict, format_parse_list

    def RAD_process(self, words, arcs, child_dict_list):
        """
        觉得鞋子的护脚踝这方面不行
        1）ATT 红色->花朵 不考虑直接放行
        2）非ATT(SBV) 鞋子->护 VOB短语作名词 脚踝->护 ATT 护->方面
        重新整理arcs关系
        """
        RAD_index_ls = [index for index, line in enumerate(arcs) if line[0] == 'RAD']
        RAD_index = RAD_index_ls[0]
        ret_words = words[:(RAD_index - 1)]

        if RAD_index == len(words) - 1:
            ret_words.append(''.join(words[(RAD_index - 1):]))
            return ret_words

        if arcs[RAD_index-1][0] == 'ATT':
            ret_words += ([''.join(words[(RAD_index - 1):(RAD_index + 2)])] + words[(RAD_index + 2):])
            return ret_words
        right_index = RAD_index + 1
        tmp_word = ''.join(words[(RAD_index - 1):right_index])
        child_dict = child_dict_list[RAD_index + 1]
        if 'VOB' in child_dict:
            if (right_index + 1) in child_dict['VOB']:
                tmp_word += ''.join(words[right_index:(right_index+2)])
                right_index += 2

        ret_words += ([tmp_word] + words[right_index:])
        return ret_words

    def main(self, sentence):
        words, postags, child_dict_list, roles_dict, format_parse_list = self.parser_main(sentence)
        arcs_head = [line[0] for line in format_parse_list]
        while 'RAD' in arcs_head:
            words = self.RAD_process(words, format_parse_list, child_dict_list)
            words, postags, child_dict_list, roles_dict, format_parse_list = self.parser_main(words, need_seg=False)
            arcs_head = [line[0] for line in format_parse_list]
        return words, postags, child_dict_list, roles_dict, format_parse_list


def test(sentence):
    words, postags, child_dict_list, roles_dict, format_parse_list = parse.main(sentence)
    # print(json.dumps(map(lambda x: x.decode('utf-8'), words), ensure_ascii=False))
    print(words)
    print(len(words))
    print(postags, len(postags))
    print(child_dict_list, len(child_dict_list))
    print(roles_dict)
    for index, line in enumerate(format_parse_list):
        # line = [str(x).decode('utf-8') for x in line]
        line = [str(index)] + line
        print("%-2s : %-5s %-8s %-5s %-5s %-10s %-5s %-5s" % (tuple(line)))
    print(len(format_parse_list))


if __name__ == '__main__':
    parse = LtpParser()

    sentence1 = u"颜值也很高穿上它就是球场最靓的仔"
    test(sentence1)
