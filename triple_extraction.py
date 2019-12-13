#!/usr/bin/env python3
# coding: utf-8
# File: triple_extraction.py
from sentence_parser import *
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
# import xlsxwriter

LABEL_DICT = os.path.abspath("new_ltp_label.txt")


class TripleExtractor:
    def __init__(self):
        self.parser = LtpParser()
        self.label_dict = self.load_dict(LABEL_DICT)

    def load_dict(self, fp):
        ret = []
        with open(fp, 'r') as f:
            tmp = f.readline()
            while tmp:
                # items = tmp.replace('\n', '').decode('utf-8')
                items = items.rsplit(' ', 1)
                if len(items) == 1:
                    ret.append(items[0])
                else:
                    ret.append((items[0], items[1]))
                tmp = f.readline()
        if type(ret[0]) is not tuple:
            ret = list(set(ret))
        else:
            ret = dict(ret)
        return ret

    """文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识"""
    def split_sents(self, content):
        # if not isinstance(content, unicode):
        #     content = content.decode('utf-8')
        content = re.sub(u'\<.[a-zA-Z0-9\=\"\:\.\?/\<\>  ]*\>', '\n', content)
        return [sentence.encode('utf-8') for sentence in re.split(u'[？\?！\!。；;\n\t ]', content) if sentence]

    """利用语义角色标注,直接获取主谓宾三元组,基于A0,A1,A2"""
    def ruler1(self, words, postags, roles_dict, role_index, child_dict_list):
        v = words[role_index]
        role_info = roles_dict[role_index]
        if 'A0' in role_info.keys() and 'A1' in role_info.keys() and v != '知道':
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2]+1) if
                     postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2]+1) if
                     postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if 'ADV' in child_dict_list[role_index]:
                ADV_index = role_index
                adv = self.ADV_complete(ADV_index, words, child_dict_list)
            else:
                adv = ''
            if s and o:
                return '1', (s, v, o, adv, "A0_A1_ADV")
        # elif 'A0' in role_info:
        #     s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
        #                  postags[word_index][0] not in ['w', 'u', 'x']])
        #     if s:
        #         return '2', [s, v]
        # elif 'A1' in role_info:
        #     o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2]+1) if
        #                  postags[word_index][0] not in ['w', 'u', 'x']])
        #     return '3', [v, o]
        return '4', ()

    # 主谓宾
    def SVO(self, index, words, postags, child_dict_list, child_dict):
        if 'SBV' in child_dict and 'VOB' in child_dict and postags[index] != 'd':
            if index - 1 >= 0:
                if words[index] == '的':
                    return None
            r = words[index]
            e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
            e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            if "ADV" in child_dict_list[index]:
                ADV_index = index
                adv = self.ADV_complete(ADV_index, words, child_dict_list)
            else:
                adv = ''
            ret = (e1, r, e2, adv, "SVO_ADV")
        else:
            ret = None
        return ret

    # 定语后置，动宾关系
    def ATT_VOB(self, index, words, postags, arcs, child_dict_list, child_dict):
        """定语后置，动宾关系"""
        relation = arcs[index][0]
        head = arcs[index][2]
        if relation == 'ATT':
            if 'VOB' in child_dict:
                e1 = self.complete_e(words, postags, child_dict_list, head - 1)
                r = words[index]
                e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                temp_string = r + e2
                if temp_string == e1[:len(temp_string)]:
                    e1 = e1[len(temp_string):]
                if temp_string not in e1:
                    ret = (e1, r, e2, "ATT_VOB")
                    return ret
        return None

    # 含有介宾关系的主谓动补关系
    def SBV_CMP(self, index, words, postags, child_dict_list, child_dict):
        """含有介宾关系的主谓动补关系"""
        ret = None
        if 'SBV' in child_dict and 'CMP' in child_dict:
            e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
            cmp_index = child_dict['CMP'][0]
            r = words[index] + words[cmp_index]
            if 'POB' in child_dict_list[cmp_index]:
                e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                ret = (e1, r, e2, 'SBV_CMP')
        return ret

    # 不考虑介宾关系的主谓动补关系
    def SBV_CMP1(self, index, words, child_dict_list, child_dict):
        """主语缺失，CMP动补结构"""
        ret = None
        if 'SBV' in child_dict and 'CMP' in child_dict:
            CMP_index_ls = child_dict['CMP']
            e1 = words[child_dict['SBV'][0]]
            cmp_index = child_dict['CMP'][0]
            r = words[index] + words[cmp_index]
            ADVs = child_dict_list[cmp_index].get('ADV', [])
            ADVs = '::'.join([words[x] for x in ADVs])
            ret = (e1, r, ADVs, 'SBV_CMP_ADV')

        return ret

    """三元组抽取主函数"""
    def rule_main(self, words, postags, child_dict_list, arcs, roles_dict):
        svos = []
        for index in range(len(postags)):
            tmp = 1
            arcs_head = list(map(lambda x: x[0], arcs))
            if 'RAD' in arcs_head or 'LAD' in arcs_head:
                break
            # 先借助语义角色标注的结果，进行三元组抽取
            if index in roles_dict:
                flag, triple = self.ruler1(words, postags, roles_dict, index, child_dict_list)
                if flag == '1':
                    svos.append(triple)
                    tmp = 0
            if tmp == 1:
                # 如果语义角色标记为空，则使用依存句法进行抽取
                if postags[index]:
                    # 抽取以谓词为中心的事实三元组
                    child_dict = child_dict_list[index]
                    # 主谓宾
                    res = self.SVO(index, words, postags, child_dict_list, child_dict)
                    if res:
                        svos.append(res)
                    # 定语后置，动宾关系
                    res = self.ATT_VOB(index, words, postags, arcs, child_dict_list, child_dict)
                    if res:
                        svos.append(res)
                    # 主语缺失情况下含有介宾关系的主谓动补关系（容易混淆，放在最后做，不能随index一个个做）
                    # res = self.SBV_CMP(index, words, postags, child_dict_list, child_dict)
                    res = self.SBV_CMP1(index, words, child_dict_list, child_dict)
                    if res:
                        svos.append(res)

        method = [t[-1] for t in svos]
        if 'A0_A1_ADV' not in method or len(svos) < 2:
            res = self.CMP_post(words, postags, child_dict_list, arcs)
            if res:
                svos.extend(res)

        return svos

    # 多主体但没有被前4个规则捕获,SBV和CMP的单结构
    def CMP_post(self, words, postags, child_dict_list, arcs):
        """主语缺失，不含SBV结构，仅有CMP动补结构"""
        ret = []
        arc_relations = list(map(lambda x: x[0], arcs))
        # {'动词补语'(index_str):'ADV'(str)s}
        res = self.CMP_ADV(words, postags, arcs, arc_relations, child_dict_list)
        if res:
            ret.extend(res)

        res = self.SBV_ADV(words, arcs, arc_relations, child_dict_list)
        if res:
            ret.extend(res)

        res = self.VOB_ADV(words, arcs, arc_relations, child_dict_list)
        if res:
            ret.extend(res)

        res = self.ATT_ADV(words, arcs, arc_relations, child_dict_list)
        if res:
            ret.extend(res)
        return ret

    def CMP_ADV(self, words, postags, arcs, arc_relations, child_dict_list):
        """
        缺失主语的CMP动补结构
        :param words:
        :param postags:
        :param arcs:
        :param arc_relations:
        :param child_dict_list:
        :return:
        """
        relations = []
        ret = []
        if 'CMP' in arc_relations and 'n' not in postags:
            # 先找有动补结构的ADV
            CMPs = [(index, words[index]) for index, _ in enumerate(arc_relations) if _ == 'CMP']
            for i, j in CMPs:
                ADV_index = i
                ADVs = self.ADV_complete(ADV_index, words, child_dict_list)
                name = arcs[i][-3]
                relations.append((name, j, ADVs))
                del ADVs
            del CMPs
            # 输出
            if len(relations) > 0:
                for v, a, adv in relations:
                    ret.append(('AUTO', v, a, adv, 'CMP_ADV'))
        return ret

    def SBV_ADV(self, words, arcs, arc_relations, child_dict_list):
        """
        缺失谓语的SBV主谓结构（其实是定语结构）
        :param words:
        :param arcs:
        :param arc_relations:
        :param child_dict_list:
        :return:
        """
        relations = []
        ret = []
        if 'SBV' in arc_relations:
            # 先找有动补结构的ADV
            SBVs = [(index, words[index]) for index, _ in enumerate(arc_relations) if _ == 'SBV']
            for i, j in SBVs:
                # 词性必须是va
                tags = arcs[i][-4] + arcs[i][-1]
                if tags != 'va' and tags != 'na':
                    continue
                ADV_index = arcs[i][-2]
                ADVs = self.ADV_complete(ADV_index, words, child_dict_list)
                name = arcs[i][-3]
                name0_ls = [i] + child_dict_list[i].get('ATT', [])
                name0_ls = sorted(name0_ls)
                name0 = ''.join([words[x] for x in name0_ls])
                relations.append((name0, name, ADVs))
                del ADVs
            del SBVs
            # 输出
            if len(relations) > 0:
                for s, v, adv in relations:
                    ret.append((s, 'AUTO', v, adv, 'SBV_ADV'))
        return ret

    def VOB_ADV(self, words, arcs, arc_relations, child_dict_list):
        """
        缺失主语的VOB动宾结构
        :param words:
        :param arcs:
        :param arc_relations:
        :param child_dict_list:
        :return:
        """
        relations = []
        ret = []
        if 'VOB' in arc_relations:
            # 先找有动宾结构的ADV
            VOBs = [(index, words[index]) for index, _ in enumerate(arc_relations) if _ == 'VOB']
            for i, j in VOBs:
                # 词性必须是va
                tags = arcs[i][-4] + arcs[i][-1]
                if tags != 'nv' and tags != 'av':
                    continue
                ADV_index = arcs[i][-2]
                ADVs = self.ADV_complete(ADV_index, words, child_dict_list)
                name = arcs[i][-3]
                relations.append((name, j, ADVs))
                # relations[name + j] = ADVs
                del ADVs
            del VOBs
            # 输出
            if len(relations) > 0:
                for v, b, adv in relations:
                    ret.append(('AUTO', v, b, adv, 'VOB_ADV'))
        return ret

    def ATT_ADV(self, words, arcs, arc_relations, child_dict_list):
        """
        谓语缺失的ATT定语结构，与SBV_ADV互为补充
        :param words:
        :param arcs:
        :param arc_relations:
        :param child_dict_list:
        :return:
        """
        relations = []
        ret = []
        if 'ATT' in arc_relations:
            # 有SVO的情况而且句子不长的话， 需要关闭
            # 先找有动补结构的ADV
            ATTs = [(index, words[index]) for index, _ in enumerate(arc_relations) if _ == 'ATT']
            for i, j in ATTs:
                # 词性必须是va
                tags = arcs[i][-4] + arcs[i][-1]
                if tags != 'an':
                    continue
                ADV_index = arcs[i][-5]
                ADVs = self.ADV_complete(ADV_index, words, child_dict_list)
                name = arcs[i][-3]
                relations.append((name, j, ADVs))
                # relations[name + j] = ADVs
                del ADVs
            del ATTs
            # 输出
            if len(relations) > 0:
                for n, a, adv in relations:
                    ret.append((n, 'AUTO', a, adv, 'ATT_ADV'))
        return ret

    def ADV_complete(self, ADV_index, words, child_dict_list):
        """
        ADV补足 主要靠POB 感觉没有比boost舒服
        :param ADV_index: 谓语所在位置，或者需要提供副词描述的词的位置
        :param words: 切词list
        :param child_dict_list: dict
        :return: ADV_str
        """
        ADVs = child_dict_list[ADV_index].get('ADV', [])
        ADV_words = []
        words_now = words[:]
        for index, x in enumerate(ADVs):
            # 补足诸如"不是", "不会", "没有"
            ADV2s = child_dict_list[x].get('ADV', [])
            words_index = sorted([x] + ADV2s)
            words_now[x] = ''.join([words_now[i] for i in words_index])
            POBs = child_dict_list[x].get('POB', [])
            words_index = sorted([x] + POBs)
            tmp_ADV_words = ''.join([words_now[i] for i in words_index])
            ADV_words.append(tmp_ADV_words)
        return ''.join(ADV_words)

    # 对找出的主语或者宾语进行扩展
    def complete_e(self, words, postags, child_dict_list, word_index):
        """对找出的主语或者宾语进行扩展"""
        child_dict = child_dict_list[word_index]
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        return prefix + words[word_index] + postfix

    """程序主控函数"""
    def triples_main(self, content):
        sentences = self.split_sents(content)
        svos = []
        with tqdm(total=len(sentences)) as pbar:
            for sentence in sentences:
                # if not isinstance(sentence, unicode):
                #     sentence = sentence.decode('utf-8')
                if len(sentence) > 55:
                    continue
                words, postags, child_dict_list, roles_dict, arcs = self.parser.main(sentence)
                svo = self.rule_main(words, postags, child_dict_list, arcs, roles_dict)
                # print(json.dumps(svo, ensure_ascii=False))
                # svo = [u'|'.join([i if isinstance(i, unicode) else i.decode('utf-8') for i in son]) for son in svo]
                svo = [u'|'.join(son) for son in svo]
                svos.append((sentence, svo))
                pbar.update(1)

        return svos

    def get_tripples(self, content):
        sentences = self.split_sents(content)
        svos = []
        for sentence in sentences:
            # if not isinstance(sentence, unicode):
            #     sentence = sentence.decode('utf-8')
            if len(sentence) > 20:
                continue
            words, postags, child_dict_list, roles_dict, arcs = self.parser.parser_main(sentence)
            svo = self.rule_main(words, postags, child_dict_list, arcs, roles_dict)
            # svo = [u'|'.join([i if isinstance(i, unicode) else i.decode('utf-8') for i in son]) for son in svo]
            svo = [u'|'.join(son) for son in svo]
            svos.append((sentence, svo))
        return svos

    def label_hit(self, tripples):
        """

        :param trpples: list of tuple
        :return: labels
        """
        method = [t[-1] for t in tripples]
        ret = [None] * len(tripples)
        for index, t in enumerate(tripples):
            pass
        # prepare_content = [ for t in tripples]


def test(content):
    """测试"""
    extractor = TripleExtractor()
    svos = extractor.triples_main(content)
    print('svos:')
    print(json.dumps(svos, ensure_ascii=False))


def test_batch(fp):
    if fp.endswith('.xlsx'):
        df = pd.read_excel(fp, encoding='utf-8')
    elif fp.endswith('.csv'):
        df = pd.read_csv(fp, encoding='utf-8')

    # r = np.random.randint(0, df.shape[0], min(2000, df.shape[0]))
    # df = df.loc[r, :]
    extractor = TripleExtractor()
    sentences = df[u'正文'].tolist()
    tripple_ls = []
    sentence_ls = []
    with tqdm(total=len(sentences)) as pbar:
        for content in sentences:
            try:
                svos = extractor.get_tripples(content)
            except:
                pbar.update(1)
                continue
            tmp_sentence_ls = []
            tmp_trip_ls = []
            for tmp_s, tmp_trips in svos:
                if len(tmp_trips) > 0:
                    tmp_sentence_ls.extend([tmp_s]*len(tmp_trips))
                    tmp_trip_ls.extend(tmp_trips)
            sentence_ls.extend(tmp_sentence_ls)
            tripple_ls.extend(tmp_trip_ls)
            pbar.update(1)
    out_df = pd.DataFrame(zip(sentence_ls, tripple_ls), columns=['sentences', 'tripples'])
    out_df.to_excel("tripple_test.xlsx", encoding='utf-8', index=None, engine='xlsxwriter')


if __name__ == '__main__':
    s = '穿着很舒服，样子也喜欢，在虎扑看到的推荐店铺，相信是正品，以后会继续支持的?'
    s = '给女朋友买的 上脚很好看，很舒服的一双鞋子，我也想买一双呢，经典的黑白款，舒服又好看，再配上媳妇的美腿，不说了我去撸撸会儿'
    s = '匡威历史比万斯来的悠久。 同样为美国的品牌，各自的崇拜者都很多 匡威较早进入中国，所以被国人认知的也比万斯多。 所以相对来说，匡威的品牌影响力还是比万斯来的大的。 喜欢帆布鞋的头把交椅无人撼动。不开胶不Vans。'
    s = '好看，穿着挺舒服的，百搭，什么裤子都能搭。就是夏天穿有点热。水晶底，非常的耐磨。反正就是很不错，挺好的。'
    s = '后掌的加大zoom气垫终于有感觉了（我个人体重130左右）前掌稍硬为给提速让步嘛可以理解'
    s = '感觉并不是很适合实战打球。'
    # s = '包裹感并没有些许问题'
    # s = "不打外场耐磨对我自己来说并不太重要"
    # s = "缓震好启动不行"
    s = "缓震好但启动不行。真的炒鸡炒鸡好看哦。抓地力各方面一般。耐磨一般，不是很推荐长期水泥地"
    s = "过了这么久克莱的鞋终于来到了第四代"
    s = "不会有鞋内漂移情况发生。Foam不如李宁的云减震软弹。鞋子有点偏小，得买大一个码数，质量特别。颜值也很高穿上它就是球场最靓的仔"

    # s = '<p>我反正不是第一个拿到鞋子的，也不追求什么首发</p><p>Kyrie 4 BHM是继2代 BHM后，我个人最中意的一双Kyrie<br>Kyrie 4 BHM的质感很好，899的价格来说可能要比打折后的AJ31的质感还要出色，配色也不错，可惜做工有点拖后腿<br>下面也许是很多人对Kyrie 4最关注的地方，之前简单试穿过黑白配色的，中底厚了，场地感明显要比2,3差，脚感和缓震以及舒适度要略微胜于1代，这也是在向市场妥协，但是千万不要过度迷信于Cushlon，就是加了点橡胶的Phylon，虽然烧制工艺上要比李宁云出色，在缓震性能上还是无法达到质变的程度，穿过渣科7力量系统的应该多少会有所体会，我最近天天在穿火花3，P-Motive真的是太爽了，Cushlon真的还是无法达到软弹的境界，要我打球，我还是情愿穿Kyrie 3换双鞋垫。下面依旧是懒得拍照，几张手机随便瞎拍<img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_7643f1f7ca5ef88fad7ee7cd4c865c9f.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x1320_084ed499a0147543e92d0868680de58a.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_45063fada345f699d01b1ed697ac52fb.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_f26c7a6ed8ed27824b05e8372a8cb9eb.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_323ea22c07813464293ec9f9ee6d246c.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_66e0b57b876a3fd45326cc325433b7b5.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_2e345aefa545f56c513de7df159e354a.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_15704fe8f7bca8594ed7888eeee40bd4.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_1a187da63b37aa6327fd8f8e2c322fe9.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_c4f7066c7dc85d03b76df5d608264119.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x428_c5c457c1e77c75ff665fd2ced0be5498.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/750x1320_d2ea827f0a5fb7b8989669174b224e99.jpeg?imageMogr2/format/jpg%7cimageView2/2/w/700/interlace/1" class="userImg"><br><img src="http://shihuo.hupucdn.com/ucditor/20180117/1600x902_49da7ae88f270b8d6b274f6945c76f58.jpeg?imageView2/2/w/700/interlace/1%7cimageMogr2/format/jpg" class="userImg"><br></p><widget-equip data-goodsid="64762" data-styleid="129806" data-supplierid="" data-title="Nike Kyrie 4" data-price="898.00" data-currency="¥" data-img="//shihuo.hupucdn.com/trade/reposition/2018-01-07/56edbf471b8d3d15de5c4830f00b97f4.png"></widget-equip><p><br></p>'
    # s = "感觉没有比boost舒服。鞋子前掌包裹性强，但是后掌有点不跟鞋"
    # s = "包裹性很高。pk鞋面透气性也很好。缓震也算是不错。的zoom缓震还算是不错"
    # s = "感觉唯一的缺点就是鞋底不耐磨。特别软但是启动又特别快。但就是鞋底不是很耐磨。纯论缓震非常棒。正码包裹肯定没得说"
    # s = "穿上刚刚好也不压脚背。包裹很不错。非常好看的一款配色，希望海沃德越打越好。觉得鞋子的护脚踝这方面不行"
    # s = "后面的塑料三角支撑做工很粗糙"
    # s = "调教的不错前掌启动不拖沓"
    # s = "防震也非常好就是要踩开防震"
    # s = "缓震好但启动不行。启动也不拖沓。调教的不错前掌启动不拖沓。防震也非常好就是要踩开防震。"
    test(s)
