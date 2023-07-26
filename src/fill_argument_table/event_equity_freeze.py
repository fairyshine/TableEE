#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import os
import re
import copy

import numpy as np
from tqdm import tqdm


def sort_pred_list(pred_list):
    """
    使用冒泡排序，按照在文本中的出现次序排列预测结果
    """
    def pred_lt(a, b):
        if a[0] < b[0]:
            return True
        elif a[0] == b[0]:
            if a[1] < b[1]:
                return True
            else:
                return False
        else:
            return False
    for i in range(len(pred_list)-1):
        stop = True
        for j in range(len(pred_list)-i-1):
            if not pred_lt(pred_list[j], pred_list[j+1]):
                pred_list[j], pred_list[j+1] = pred_list[j+1], pred_list[j]
                stop = False
        if stop:
            break
    return pred_list


def group_pred_list(pred_list):
    """
    将预测结果按句子id分组，并按序排列
    """
    sent2pred = {}
    for pred in pred_list:
        sent_id = pred[0]
        if sent_id not in sent2pred.keys():
            sent2pred[sent_id] = [pred]
        else:
            sent2pred[sent_id].append(pred)
    ret = sorted(sent2pred.items(), key=lambda x: x[0], reverse=False)
    ret = [x[1] for x in ret]
    return ret


def get_pred_list(event_role_pred, original, candidate_role):
    pred_list = []
    for sent_id, span_list in event_role_pred:
        if len(span_list) == 0:
            continue
        cur_sent = original[sent_id]
        for span in span_list:
            start, end, role = span

            if role not in candidate_role:
                continue
            entity = cur_sent[start:end+1]
            pred_list.append([sent_id, start, end, role, entity])
    pred_list = sort_pred_list(pred_list)
    pred_list = group_pred_list(pred_list)
    return pred_list


def slim_answer(one_answer):
    """
    去除答案中值为None的要素
    """
    ret = {}
    for role, entity in one_answer.items():
        if entity is not None:
            ret[role] = entity
    return ret


def is_contained(ans1, ans2):
    # 判断ans1是否包含ans2
    # 对于ans2中的每个key，检查ans1中是否有，且是否相等
    if len(ans1) >= len(ans2):
        for k in ans2.keys():
            if k not in ans1.keys():
                return False
            else:
                if ans1[k] != ans2[k]:
                    return False
    else:
        return False
    return True


def is_merged(ans1, ans2):
    # 判断ans1是否可以与ans2合并
    # 如果ans1、ans2有相同的key但值不一样，那么就不可以合并
    for k in ans2.keys():
        if k == 'event_type':
            if ans1[k] != ans2[k]:
                return False
            continue

        if k not in ans1.keys():
            continue
        else:
            if ans1[k] != ans2[k]:
                return False
    return True


def del_and_add(one_answer, all_answer):
    del_idx = []
    add_flag = True
    for idx, sub_ans in enumerate(all_answer):
        if is_contained(sub_ans, one_answer):
            add_flag = False
            break
        elif is_contained(one_answer, sub_ans):
            del_idx.append(idx)
    if len(del_idx) != 0:
        new_all_answer = []
        for idx, sub_ans in enumerate(all_answer):
            if idx in del_idx:
                continue
            new_all_answer.append(sub_ans)
        all_answer = new_all_answer
    if add_flag is True:
        all_answer.append(one_answer)
    return all_answer


def merge_answer(all_answer):
    if len(all_answer) <= 1:
        return all_answer
    new_all_answer = [all_answer[0]]
    all_answer = all_answer[1:]
    while len(all_answer) > 0:
        ans1 = new_all_answer.pop()
        ans2 = all_answer.pop(0)
        if is_merged(ans1, ans2):
            merged_ans = {**ans1, **ans2}
            new_all_answer.append(merged_ans)
        else:
            new_all_answer.append(ans1)
            new_all_answer.append(ans2)
    return new_all_answer


def clean_role_list(role_list):
    new_role_list = []
    for sub_role_list in role_list:
        tmp = []
        for sub_role in sub_role_list:
            entity = sub_role[4]
            if len(entity) == 1:
                continue
            tmp.append(sub_role)
        if len(tmp) != 0:
            new_role_list.append(tmp)
    return new_role_list


def flat_role_list(role_list):
    flat_list = []
    for cur_role_pred in role_list:
        for one_role_pred in cur_role_pred:
            flat_list.append(one_role_pred)
    return flat_list


def extract_redundant_roles(sents):
    redundant_roles = []
    regex = r"流通股\s*(\d[,，\d]*\d)\s*股"
    for s_idx, sent in enumerate(sents):
        for match in re.finditer(regex, sent):
            redundant_roles.append([s_idx, match.span(1)[0], match.span(1)[
                                   1] - 1, "冻结金额", match.group(1)])
    return redundant_roles


def merge_role_list(l1, l2):
    all_ = [*l1, *l2]
    ans = []
    unique = set()
    for l in all_:
        s = "{}-{}-{}-{}-{}".format(*l)
        if s not in unique:
            ans.append(l)
            unique.add(s)
    ans = sorted(ans, key=lambda x: (x[0], x[1], x[2]))
    return ans


def remove_role_list(ori, remain_remove):
    ans = []
    rem = set()
    for r in remain_remove:
        s = "{}-{}-{}-{}-{}".format(*r)
        rem.add(s)
    for r in ori:
        s = "{}-{}-{}-{}-{}".format(*r)
        if s not in rem:
            ans.append(r)
    ans = sorted(ans, key=lambda x: (x[0], x[1], x[2]))
    return ans


def unique_role_list(rl):
    unique = set()
    ans = []
    rl = sorted(rl, key=lambda x: (x[0], x[1], x[2]))
    for r in rl:
        s = "{}-{}".format(r[-2], r[-1])
        if s not in unique:
            ans.append(r)
            unique.add(s)
    return ans


def unique_money_list(rl):
    unique = set()
    ans = []
    rl = sorted(rl, key=lambda x: (x[0], x[1], x[2]))
    for r in rl:
        if r[-2] != "冻结金额":
            ans.append(r)
            continue
        s = "{}-{}".format(r[-2], r[-1])
        if s not in unique:
            ans.append(r)
            unique.add(s)
    return ans


def clean_adj_rep_role(rl):
    def role_eq(r1, r2):
        return r1[-2] == r2[-2] and r1[-1] == r2[-1]

    taboo = set()
    cleaned = []
    for i in range(len(rl) - 1):
        if i not in taboo:
            cleaned.append(rl[i])
        if role_eq(rl[i], rl[i + 1]):
            taboo.add(i + 1)
    return cleaned


def money_normalisation(string):
    if "万" in string:
        string = string.replace("万", "0000")
    if "," in string:
        string = string.replace(",", "")
    if "，" in string:
        string = string.replace("，", "")
    return string


def extract_from_chunk_bak(role_list, answer_table, valid_sent):
    def flat_role_list(role_list):
        flat_list = []
        for cur_role_pred in role_list:
            for one_role_pred in cur_role_pred:
                flat_list.append(one_role_pred)
        return flat_list

    all_answer = []
    flat_list = flat_role_list(role_list)
    one_answer = copy.deepcopy(answer_table)

    for i, one_role_pred in enumerate(flat_list):
        pred_role = one_role_pred[3]
        if pred_role == '冻结金额':
            one_answer[pred_role] = one_role_pred[-1]
            for j in range(i, -1, -1):
                if flat_list[j][3] == '被冻结股东':
                    one_answer['被冻结股东'] = flat_list[j][-1]
                    break
            start_time = False
            end_time = False
            for j in range(i, len(flat_list)):
                if flat_list[j][3] == '冻结开始日期' and not start_time:
                    one_answer['冻结开始日期'] = flat_list[j][-1]
                    start_time = True
                if flat_list[j][3] == '冻结结束日期' and not end_time:
                    if j-i <= 3:
                        one_answer['冻结结束日期'] = flat_list[j][-1]
                    end_time = True
                if start_time and end_time:
                    break
            one_answer = slim_answer(one_answer)
            all_answer.append(one_answer)
            one_answer = copy.deepcopy(answer_table)
    return all_answer


def equity_freeze_bak(role_pred, original, sentences, doc_id, events=None):
    role_list = ['被冻结股东', '冻结金额', '冻结开始日期', '冻结结束日期']
    answer_table = {'event_type': '股权冻结'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))
    all_answer = []

    if len(pred_list) == 0:
        one_answer = copy.deepcopy(answer_table)
        one_answer = slim_answer(one_answer)
        all_answer.append(one_answer)
    else:
        role_list = clean_role_list(pred_list)
        keywords = ['冻结', '被冻结']
        valid_sent = []
        for i, sent in enumerate(original):
            flag = any(word in sent for word in keywords)
            valid_sent.append(flag)

        sub_answer = extract_from_chunk(role_list, answer_table, valid_sent)
        for one_answer in sub_answer:
            if len(all_answer) == 0:
                all_answer.append(one_answer)
            else:
                all_answer = del_and_add(one_answer, all_answer)
        all_answer = merge_answer(all_answer)
        new_all_answer = []
        for one_answer in all_answer:

            if len(one_answer) <= 3:
                continue
            new_all_answer.append(one_answer)

        all_answer = new_all_answer
    return all_answer


class Distance(object):
    def __init__(self, sent_lens):
        self.sent_lens = sent_lens
        self.distance = -1
        self.money_pos = -1
        self.shareholder_pos = -1
        self.start_datetime_pos = -1
        self.end_datetime_pos = -1

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key in ['shareholder_pos', "start_datetime_pos", "end_datetime_pos"]:
            if value == -1:
                self.__dict__[key] = 0
            else:
                for k in ['shareholder_pos', "start_datetime_pos", "end_datetime_pos"]:
                    self.distance += abs(self.money_pos - getattr(self, k))

    def get_abs_pos(self, sent_idx, pos_idx):
        if sent_idx == 0:
            return pos_idx
        elif sent_idx > 0 and sent_idx < len(self.sent_lens):
            abs_pos = 0
            for i in range(len(self.sent_lens)):
                if i < sent_idx:
                    abs_pos += self.sent_lens[i]
                elif i == sent_idx:
                    abs_pos += pos_idx
                    break
            return abs_pos
        else:
            raise ValueError


def extract_from_chunk(flat_list, answer_table, sents):
    sent_lens = [len(x) for x in sents]
    money2answer = {}
    all_answer = []
    one_answer = copy.deepcopy(answer_table)

    role2flag = {}
    for role in flat_list:
        role2flag["{}-{}-{}-{}-{}".format(*role)] = False

    for i, one_role_pred in enumerate(flat_list):
        pred_role = one_role_pred[3]
        if pred_role == '冻结金额':
            distance = Distance(sent_lens)
            distance.money_pos = distance.get_abs_pos(*one_role_pred[0:2])
            one_answer[pred_role] = one_role_pred[-1]
            if role2flag["{}-{}-{}-{}-{}".format(*one_role_pred)] == False:
                role2flag["{}-{}-{}-{}-{}".format(*one_role_pred)] = True
            else:
                continue
            for j in range(i, -1, -1):
                if flat_list[j][3] == '被冻结股东':
                    one_answer['被冻结股东'] = flat_list[j][-1]
                    distance.shareholder_pos = distance.get_abs_pos(
                        *flat_list[j][0:2])
                    break
            start_time = False
            skip_start_time = False
            end_time = False
            for j in range(i, len(flat_list)):
                if flat_list[j][3] == '冻结开始日期' and not start_time and not role2flag["{}-{}-{}-{}-{}".format(*flat_list[j])]:
                    one_answer['冻结开始日期'] = flat_list[j][-1]
                    distance.start_datetime_pos = distance.get_abs_pos(
                        *flat_list[j][0:2])
                    start_time = True
                    role2flag["{}-{}-{}-{}-{}".format(*flat_list[j])] = True
                    if j + 1 < len(flat_list) and flat_list[j + 1][3] == '冻结结束日期':
                        one_answer['冻结结束日期'] = flat_list[j + 1][-1]
                        distance.end_datetime_pos = distance.get_abs_pos(
                            *flat_list[j + 1][0:2])
                        role2flag["{}-{}-{}-{}-{}".format(
                            *flat_list[j + 1])] = True
                        end_time = True
                    else:
                        break
            one_answer = slim_answer(one_answer)
            index = ""
            for ele in ['被冻结股东', '冻结金额', '冻结开始日期', '冻结结束日期']:
                index += ele
                if ele == "冻结金额":
                    index += money_normalisation(one_answer['冻结金额'])
                elif ele in one_answer:
                    index += one_answer[ele]
                elif ele not in one_answer:
                    index += "None"
            if index not in money2answer:
                money2answer[index] = []
            money2answer[index].append([distance.distance, one_answer])
            all_answer.append(one_answer)
            one_answer = copy.deepcopy(answer_table)
    aa = []
    for m in money2answer:
        al = sorted(money2answer[m], key=lambda x: x[0])
        aa.append(al[0][1])
    return aa


def equity_freeze(role_pred, original, sentences, doc_id, events=None, money_unique=False):
    role_list = ['被冻结股东', '冻结金额', '冻结开始日期', '冻结结束日期']
    answer_table = {'event_type': '股权冻结'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))
    all_answer = []

    if len(pred_list) == 0:
        one_answer = copy.deepcopy(answer_table)
        one_answer = slim_answer(one_answer)
        all_answer.append(one_answer)
    else:
        role_list = flat_role_list(clean_role_list(pred_list))
        if money_unique:
            role_list = unique_money_list(role_list)
        keywords = ['冻结', '被冻结']
        valid_sent = []
        for i, sent in enumerate(original):
            flag = any(word in sent for word in keywords)
            valid_sent.append(flag)

        sub_answer = extract_from_chunk(role_list, answer_table, original)
        for one_answer in sub_answer:
            if len(all_answer) == 0:
                all_answer.append(one_answer)
            else:
                all_answer = del_and_add(one_answer, all_answer)
        all_answer = merge_answer(all_answer)
        new_all_answer = []
        lens = [len(x) for x in all_answer]
        for one_answer in all_answer:
            if len(one_answer) <= 3 and max(lens) > 3:
                continue

            new_all_answer.append(one_answer)
        all_answer = new_all_answer

    remain_ans = copy.deepcopy(all_answer)
    return remain_ans


def simplify_and_filter_pred(event_role_pred, event_type_pred):
    """
    将`股东减持-减持的金额`格式的简化为`减持的金额`
    事件类型和预测类型不一致的预测要素去除
    """
    ret = []
    for sent_id, sub_pred in event_role_pred:
        if len(sub_pred) > 0:
            tmp = []
            for start, end, type_role in sub_pred:
                event_type, role = type_role.split('-')
                if event_type == event_type_pred:
                    tmp.append([start, end, role])
            if len(tmp) > 0:
                ret.append([sent_id, tmp])
    return ret


def generate_answer(path_src, path_des, mode=2):
    fw = open(path_des, 'w', encoding='utf-8')
    with open(path_src, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = json.loads(line.strip())
            doc_id = line['doc_id']
            sentences = line['sentences']
            event_type_pred = line['event_type_pred']
            event_role_pred = line['event_role_pred']
            event_role_pred = simplify_and_filter_pred(
                event_role_pred, event_type_pred)

            events = line['events']
            original = line['original'] if mode == 3 else sentences
            all_answer = equity_freeze(
                event_role_pred,
                original,
                sentences,
                doc_id,
                events,
            )
            meta = dict(
                doc_id=doc_id,
                events=all_answer,
            )
            json.dump(meta, fw, ensure_ascii=False)
            fw.write('\n')
    fw.close()


if __name__ == '__main__':
    if not os.path.exists('../fusion_result'):
        os.makedirs('../fusion_result')

    event_type = '股权冻结'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
