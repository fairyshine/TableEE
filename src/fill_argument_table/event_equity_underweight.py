#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import os
import re
import copy
import numpy as np


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


def post_process(all_possible_answer):
    all_answer = []
    for one_answer in all_possible_answer:
        if len(all_answer) == 0:
            all_answer.append(one_answer)
        else:
            all_answer = del_and_add(one_answer, all_answer)

    all_answer = merge_answer(all_answer)
    return all_answer


def clean_pred_list(pred_list):
    """
    去除实体长度小于等于1的事件要素
    经统计，在目前的训练数据中，没有长度小于1的要素
    """
    ret = []
    for sub_pred in pred_list:
        tmp = []
        for pred in sub_pred:
            entity = pred[-1]
            if len(entity) <= 1:
                continue
            tmp.append(pred)
        if len(tmp) > 0:
            ret.append(tmp)
    return ret


def flatten_pred_list(pred_list):
    """
    将已经分组的预测列表展开（多维->一维）
    """
    ret = []
    for sub_pred in pred_list:
        ret += sub_pred
    return ret


def find_lcseque(s1, s2):
    '''
    求两个字符串的最长公共子序列
    '''
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2)+1)] for y in range(len(s1)+1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2)+1)] for y in range(len(s1)+1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'
            elif m[p1+1][p2] > m[p1][p2+1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2]
                d[p1+1][p2+1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]
                d[p1+1][p2+1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)


def cal_local_distance(sub_pred):
    """
    计算同一段中事件要素之间的距离
    """
    matrix = np.zeros(shape=[len(sub_pred), len(sub_pred)], dtype=np.int32)
    for i, pred in enumerate(sub_pred):
        for j in range(i, len(sub_pred)):
            if i == j:
                matrix[i][j] = 0
            else:
                cur_pred = sub_pred[j]
                matrix[i][j] = cur_pred[1]-pred[2]
                matrix[j][i] = matrix[i][j]
    return matrix


def cal_global_distance(flat_pred_list, sentences):
    """
    计算篇章文本中事件要素之间的距离
    """
    seq_lens = list(map(len, sentences))
    matrix = np.zeros(shape=[len(flat_pred_list),
                             len(flat_pred_list)], dtype=np.int32)
    for i, pred in enumerate(flat_pred_list):
        for j in range(i, len(flat_pred_list)):
            if i == j:
                matrix[i][j] = 0
            else:
                cur_pred = flat_pred_list[j]
                if pred[0] == cur_pred[0]:
                    matrix[i][j] = cur_pred[1]-pred[2]
                    matrix[j][i] = matrix[i][j]
                else:
                    gap_dis = sum(seq_lens[pred[0]+1:cur_pred[0]])
                    matrix[i][j] = cur_pred[1] + (seq_lens[pred[0]]-1-pred[2])\
                        + gap_dis
                    matrix[j][i] = matrix[i][j]
    return matrix


def flat_role_list(role_list):
    flat_list = []
    for cur_role_pred in role_list:
        for one_role_pred in cur_role_pred:
            flat_list.append(one_role_pred)
    return flat_list


def extract_from_chunk(role_list, answer_table, valid_sent):

    all_answer = []
    one_answer = copy.deepcopy(answer_table)
    flat_list = flat_role_list(role_list)

    for i, one_role_pred in enumerate(flat_list):
        sent_id = flat_list[i][0]
        if valid_sent[sent_id] is False:
            continue
        pred_role = one_role_pred[3]
        if pred_role == '减持金额':
            one_answer[pred_role] = one_role_pred[-1]
            for j in range(i, -1, -1):
                if flat_list[j][3] == '减持开始日期':
                    one_answer['减持开始日期'] = flat_list[j][-1]
                    break
            for j in range(i, -1, -1):
                if flat_list[j][3] == '减持的股东':
                    one_answer['减持的股东'] = flat_list[j][-1]
                    break
            one_answer = slim_answer(one_answer)
            all_answer.append(one_answer)
            one_answer = copy.deepcopy(answer_table)
    return all_answer


def intial_predct(pred_list, original, answer_table):

    keywords = ['减持']
    valid_sent = []
    for i, sent in enumerate(original):
        flag = any(word in sent for word in keywords)
        valid_sent.append(flag)

    all_answer = extract_from_chunk(pred_list, answer_table, valid_sent)

    return all_answer


def equity_underweight(role_pred, original, sentences, doc_id, events=None):
    role_list = ['减持的股东', '减持金额', '减持开始日期']
    answer_table = {'event_type': '股东减持'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))

    all_answer = []
    if len(pred_list) == 0:
        one_answer = copy.deepcopy(answer_table)
        one_answer = slim_answer(one_answer)
        all_answer.append(one_answer)
    else:
        pred_list = clean_pred_list(pred_list)
        flat_list = flat_role_list(pred_list)

        seq_lens = list(map(len, sentences))
        global_pred_list = []
        for pred in flat_list:
            sent_id, start, end, role, entity = pred
            shift = sum(seq_lens[:sent_id]) + sent_id
            ns = shift + start
            ne = shift + end
            global_pred_list.append((ns, ne, role, entity))

        text = ' '.join(sentences)
        start = [x for x in re.finditer(r'股东名称|减持股东|股东姓名', text)]
        end = [x for x in re.finditer(r'合\s?计|减持前持有股份|二、', text)]

        # component_verify = True
        # 根据合计检验潜在减持条数，只匹配到减持前持有股份时，可能仅一条减持事件。
        # 金额和时间,和公司全称在表格中分析后，需要返回原文进行再次匹配。

        if len(start) != 0:
            start_index = start[0].span()[0]
            end_index = 0
            for end_component in end:
                if end_component.span()[0] > start_index:
                    end_index = end_component.span()[0]
                    break
            if end_index == 0:   # 可能找不到表尾，在后面需要额外处理。
                end_index = len(text)

            one_answer = copy.deepcopy(answer_table)

            intra_table_count = 0

            for one_role_pred in global_pred_list:
                if one_role_pred[0] >= start_index and one_role_pred[1] <= end_index and one_role_pred[2] == '减持金额':
                    intra_table_count = intra_table_count + 1

            if intra_table_count > 0:
                for i, one_role_pred in enumerate(global_pred_list):

                    pred_start_index = one_role_pred[0]
                    pred_end_index = one_role_pred[1]
                    pred_role = one_role_pred[2]

                    if pred_start_index >= start_index and pred_end_index <= end_index:
                        if pred_role == '减持金额':
                            one_answer[pred_role] = one_role_pred[-1]
                            for j in range(i, -1, -1):
                                if global_pred_list[j][2] == '减持开始日期':
                                    one_answer['减持开始日期'] = flat_list[j][-1]
                                    break
                            for j in range(i, -1, -1):
                                if global_pred_list[j][2] == '减持的股东':
                                    one_answer['减持的股东'] = flat_list[j][-1]
                                    break
                            one_answer = slim_answer(one_answer)
                            all_answer.append(one_answer)
                            one_answer = copy.deepcopy(answer_table)

            else:
                all_answer = intial_predct(pred_list, original, answer_table)
            all_answer = post_process(all_answer)

        else:
            pred_list = clean_pred_list(pred_list)
            keywords = ['减持']
            valid_sent = []
            for i, sent in enumerate(original):
                flag = any(word in sent for word in keywords)
                valid_sent.append(flag)

            sub_answer = extract_from_chunk(
                pred_list, answer_table, valid_sent)
            all_answer = post_process(sub_answer)

    return all_answer


def show_role_pred(role_list):
    for sub_role_list in role_list:
        print(sub_role_list)


def show_sent(sentences):
    for i, sent in enumerate(sentences):
        print('{:>2d}\t{}'.format(i, sent))


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
            all_answer = equity_underweight(
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

    event_type = '股东减持'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
