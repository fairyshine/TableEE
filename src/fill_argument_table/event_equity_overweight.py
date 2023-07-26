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
    # 过滤掉元素少于两个的答案
    # 但如果所有答案包含的要素都少于2，则提交相对来说数量最多的那个答案
    ret = []
    max_len = 0
    best_ans = None
    for one_answer in all_answer:
        ans_len = len(one_answer)
        if ans_len > max_len:
            best_ans = one_answer
            max_len = ans_len
        if len(one_answer) <= 2:
            continue
        ret.append(one_answer)
    if len(ret) == 0 and best_ans is not None:
        ret.append(best_ans)
    return ret
    # return all_answer


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


def str2float(string):
    """
    将金额转为浮点数
    """
    num_str = ''
    for c in string:
        if c >= '0' and c <= '9':
            num_str += c
        elif c == '.' and '.' not in num_str:
            num_str += c
    num = float(num_str)
    if '百' in string:
        num *= 100
    if '千' in string:
        num *= 1000
    if '万' in string:
        num *= 1e4
    if '亿' in string:
        num *= 1e8
    return num


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
    matrix = np.zeros(shape=[len(flat_pred_list), len(flat_pred_list)], dtype=np.int32)
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


def get_date_dict(flat_pred_list):
    """
    合并因日期不同而导致不同的答案
    """
    regex = r'(20[0-2]\d年)?[01]?\d月[0-3]?\d日?|20[0-2]\d\.[01]?\d\.[0-3]?\d|20[0-2]\d-[01]?\d-[0-3]?\d|20[0-2]\d/[01]?\d/[0-3]?\d'
    date_dict = {}
    date_freq = {}
    for pred in flat_pred_list:
        if pred[-2] != '增持开始日期':
            continue
        pred_date = pred[-1]
        date_freq[pred_date] = date_freq.get(pred_date, 0) + 1
        if re.fullmatch(regex, pred_date) is None:
            date_dict[pred_date] = pred_date
        else:
            if '-' in pred_date:
                value = [int(x) for x in pred_date.split('-')]
                date_dict[pred_date] = '{}年{}月{}日'.format(value[0], value[1], value[2])
            elif '.' in pred_date:
                value = [int(x) for x in pred_date.split('.')]
                date_dict[pred_date] = '{}年{}月{}日'.format(value[0], value[1], value[2])
            elif '/' in pred_date:
                value = [int(x) for x in pred_date.split('/')]
                date_dict[pred_date] = '{}年{}月{}日'.format(value[0], value[1], value[2])
            else:
                if '日' not in pred_date:
                    date_dict[pred_date] = pred_date + '日'
                else:
                    date_dict[pred_date] = pred_date
    return date_dict, date_freq


def sent2text(sentences, original, flat_pred_list):
    """
    将句子级的预测结果的下标索引变成全文的索引
    句子之间也添加空格，合并成篇章
    """
    seq_lens = list(map(len, sentences))
    global_pred_list = []
    for pred in flat_pred_list:
        sent_id, start, end, role, entity = pred
        shift = sum(seq_lens[:sent_id]) + sent_id
        ns = shift + start
        ne = shift + end
        global_pred_list.append((ns, ne, role, entity))
    text = ' '.join(sentences)
    original_text = ' '.join(original)
    sent_span = []
    for sent_id, sent in enumerate(sentences):
        shift = sum(seq_lens[:sent_id]) + sent_id
        ns = shift + 0
        ne = shift + len(sent) - 1
        sent_span.append((ns, ne))
    return text, original_text, global_pred_list, sent_span


def complete_one_answer_v1(cur_idx, one_answer, global_dis, flat_pred_list, max_dis=200):
    """
    如果文中就出现一个质押金额或没有金额，使用此方案
    """
    pred = flat_pred_list[cur_idx]
    pred_role = pred[3]
    one_answer[pred_role] = pred[-1]
    cur_dis = [(idx, x) for idx, x in enumerate(global_dis[cur_idx])]
    sorted_dis = sorted(cur_dis, key=lambda x: x[1], reverse=False)
    indices = [cur_idx]
    for idx, dis in sorted_dis:
        if dis == 0:
            continue
        if dis > max_dis:
            break
        cur_pred = flat_pred_list[idx]
        cur_role = cur_pred[3]
        cur_entity = cur_pred[4]
        if one_answer[cur_role] is None:
            one_answer[cur_role] = cur_entity
            indices.append(idx)
    total_dis = 0
    indices = sorted(indices)
    for i in range(len(indices)-1):
        total_dis += global_dis[indices[i]][indices[i+1]]
    one_answer = slim_answer(one_answer)
    return one_answer, total_dis


def generate_event_1(flat_pred_list, sentences, answer_table):
    """
    如果没有出现质押金额，就找出内联距离（要素之间距离之和）最小的一个组合
    且保证要素的数量尽可能地多
    只返回一个最佳结果
    """
    all_answer = []
    global_dis = cal_global_distance(flat_pred_list, sentences)
    best_ans = None
    min_dis = float('inf')
    for i, pred in enumerate(flat_pred_list):
        one_answer = copy.deepcopy(answer_table)
        one_answer, total_dis = complete_one_answer_v1(i, one_answer, global_dis, flat_pred_list)
        if best_ans is None:
            best_ans = one_answer
            min_dis = total_dis
        else:
            if len(one_answer) > len(best_ans):
                best_ans = one_answer
                min_dis = total_dis
            elif len(one_answer) == len(best_ans):
                if total_dis < min_dis:
                    best_ans = one_answer
                    min_dis = total_dis
    if best_ans is not None:
        all_answer.append(best_ans)
    return all_answer


def process_text(pred_list, flat_pred_list, original, sentences, answer_table, doc_id, events=None):
    text, original_text, global_pred_list, sent_span = sent2text(
        sentences, original, flat_pred_list
    )
    money = set()
    for pred in flat_pred_list:
        _, _, _, role, entity = pred
        if role == '增持金额':
            money.add(entity)

    regex = r'[一二三]?十?[一二三四五六七八九十]、|重要内容提示:'
    num_match = [x for x in re.finditer(regex, text)]
    cut_pos = [0]
    for num in num_match:
        start, _ = num.span()
        cut_pos.append(start)
    cut_pos.append(len(text))
    answer_value = []
    for k in range(len(cut_pos)-1):
        start = cut_pos[k]
        end = cut_pos[k+1]-1
        valid_pred_list = []
        pre_pred_list = []
        for pred in global_pred_list:
            pred_start = pred[0]
            pred_end = pred[1]
            if pred_start >= start and pred_end <= end:
                valid_pred_list.append(pred)
            elif pred_end < start:
                pre_pred_list.append(pred)
        for i, pred in enumerate(valid_pred_list):
            pred_start, pred_end, role, entity = pred
            if role != '增持金额':
                continue
            one_answer = copy.deepcopy(answer_table)
            one_answer['增持金额'] = entity
            first_pred_end = pred_end
            for j in range(i, -1, -1):
                cur_role = valid_pred_list[j][-2]
                cur_entity = valid_pred_list[j][-1]
                if one_answer[cur_role] is None:
                    one_answer[cur_role] = cur_entity
                    first_pred_end = valid_pred_list[j][1]
            total_dis = pred_start - first_pred_end
            one_answer = slim_answer(one_answer)
            if '万' not in one_answer['增持金额'] and \
                    '{}万'.format(one_answer['增持金额']) in money:
                one_answer['增持金额'] += '万'
            answer_value.append((one_answer, total_dis))
    if len(answer_value) == 0:
        all_answer = generate_event_1(flat_pred_list, sentences, answer_table)
    elif len(answer_value) == 1:
        all_answer = [x[0] for x in answer_value]
    else:
        all_answer = []
        # 默认答案中增持金额一样的事件为同一事件，
        # 对于增持金额相同的事件，选取要素多的、内联距离小的事件
        for i in range(len(answer_value)):
            cur_value = str2float(answer_value[i][0]['增持金额'])
            cur_dis = answer_value[i][1]
            flag = True
            for j in range(len(answer_value)):
                if i == j:
                    continue
                try: # * 修改
                    value = str2float(answer_value[j][0]['增持金额'])
                except:
                    continue
                dis = answer_value[j][1]
                if cur_value == value:
                    if len(answer_value[j][0]) > len(answer_value[i][0]):
                        flag = False
                        break
                    elif len(answer_value[j][0]) == len(answer_value[i][0]):
                        if dis < cur_dis:
                            flag = False
                            break
            if flag:
                all_answer.append(answer_value[i][0])
    date_dict, date_freq = get_date_dict(flat_pred_list)
    # 如果全文就预测出一个日期，则对那些没有增持开始日期的事件进行补充
    if len(date_freq) == 1:
        date_list = sorted(date_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, one_answer in enumerate(all_answer):
            if '增持开始日期' not in one_answer.keys():
                all_answer[idx]['增持开始日期'] = date_list[0][0]
    # 日期归一化（为了方便去重）

    for idx, one_answer in enumerate(all_answer):
        if '增持开始日期' in one_answer.keys():
            for date in date_dict.keys():
                if date == one_answer['增持开始日期']:
                    continue
                elif one_answer['增持开始日期'] in date:
                    one_answer['增持开始日期'] = date
                    break
            if date_dict[one_answer['增持开始日期']] in date_dict.keys():
                all_answer[idx]['增持开始日期'] = date_dict[one_answer['增持开始日期']]

    all_answer = post_process(all_answer)
    shareholder = {}
    for pred in global_pred_list:
        if pred[-2] == '增持的股东':
            shareholder[pred[-1]] = shareholder.get(pred[-1], 0) + 1
    # 有的答案中没有增持的股东一项，需要进行补充
    if len(shareholder) > 0:
        # 统计已有答案中增持的股东的频次
        shareholder_in_answer = {}
        for one_answer in all_answer:
            if '增持的股东' in one_answer.keys():
                t = one_answer['增持的股东']
                shareholder_in_answer[t] = shareholder_in_answer.get(t, 0) + 1
        # 优先选择已有答案中的增持的股东进行填充，其次是最高频次股东
        if len(shareholder_in_answer) > 0:
            shareholder_in_answer_list = sorted(shareholder_in_answer.items(), key=lambda x: x[1], reverse=True)
            best_candi = shareholder_in_answer_list[0][0]
        else:
            shareholder_list = sorted(shareholder.items(), key=lambda x: x[1], reverse=True)
            best_candi = shareholder_list[0][0]
        for i, one_answer in enumerate(all_answer):
            if '增持的股东' not in one_answer.keys():
                all_answer[i]['增持的股东'] = best_candi
        # 通过对比最长公共子序列的方法找全称
        for one_answer in all_answer:
            for s in shareholder.keys():
                if s == one_answer['增持的股东']:
                    continue
                lcs = find_lcseque(s, one_answer['增持的股东'])
                if lcs == one_answer['增持的股东']:
                    # 如果只是单纯的多个空格，则不认为是全称
                    if ''.join(s.split()) == one_answer['增持的股东']:
                        continue
                    one_answer['增持的股东'] = s
                    break
    return all_answer


def process_table(pred_list, flat_pred_list, original, sentences, answer_table, doc_id, table_search, events):
    text, original_text, global_pred_list, sent_span = sent2text(
        sentences, original, flat_pred_list
    )
    money = set()
    for pred in flat_pred_list:
        _, _, _, role, entity = pred
        if role == '增持金额':
            money.add(entity)
    regex = r'[一二三]?十?[一二三四五六七八九十]、|重要内容提示:'
    num_match = [x for x in re.finditer(regex, text)]
    cut_pos = [0]
    for num in num_match:
        start, _ = num.span()
        cut_pos.append(start)
    cut_pos.append(len(text))
    sum_flag = [x for x in re.finditer(r'合\s?计', text)]
    table_start = table_search.span()[1]
    table_end = table_start
    for k in range(len(cut_pos)-1):
        start = cut_pos[k]
        end = cut_pos[k+1]-1
        if start <= table_start and table_start <= end:
            table_end = end
            break
    for flag in sum_flag:
        start, end = flag.span()
        end = end - 1
        if start >= table_start and end <= table_end:
            table_end = start - 1
            break
    all_answer = []
    start = table_start
    end = table_end
    valid_pred_list = []
    for pred in global_pred_list:
        pred_start = pred[0]
        pred_end = pred[1]
        if pred_start >= start and pred_end <= end:
            valid_pred_list.append(pred)
    for i, pred in enumerate(valid_pred_list):
        role = pred[-2]
        entity = pred[-1]
        if role != '增持金额':
            continue
        one_answer = copy.deepcopy(answer_table)
        one_answer['增持金额'] = entity
        for j in range(i, -1, -1):
            cur_role = valid_pred_list[j][-2]
            cur_entity = valid_pred_list[j][-1]
            if one_answer[cur_role] is None:
                one_answer[cur_role] = cur_entity
        if '万' not in one_answer['增持金额'] and \
                '{}万'.format(one_answer['增持金额']) in money:
            one_answer['增持金额'] += '万'
        one_answer = slim_answer(one_answer)
        all_answer.append(one_answer)
    all_answer = post_process(all_answer)
    shareholder = {}
    for pred in global_pred_list:
        if pred[-2] == '增持的股东':
            shareholder[pred[-1]] = shareholder.get(pred[-1], 0) + 1

    date_dict, date_freq = get_date_dict(flat_pred_list)
    # 如果全文就预测出一个日期，则对那些没有增持开始日期的事件进行补充
    if len(date_freq) == 1:
        date_list = sorted(date_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, one_answer in enumerate(all_answer):
            if '增持开始日期' not in one_answer.keys():
                all_answer[idx]['增持开始日期'] = date_list[0][0]
    # 日期归一化（为了方便去重）
    for idx, one_answer in enumerate(all_answer):
        if '增持开始日期' in one_answer.keys():
            for date in date_dict.keys():
                if date == one_answer['增持开始日期']:
                    continue
                elif one_answer['增持开始日期'] in date:
                    one_answer['增持开始日期'] = date
                    break

    if len(shareholder) > 0:
        # 统计已有答案中增持的股东的频次
        shareholder_in_answer = {}
        for one_answer in all_answer:
            if '增持的股东' in one_answer.keys():
                t = one_answer['增持的股东']
                shareholder_in_answer[t] = shareholder_in_answer.get(t, 0) + 1
        # 优先选择已有答案中的增持的股东进行填充，其次是最高频次股东
        if len(shareholder_in_answer) > 0:
            shareholder_in_answer_list = sorted(shareholder_in_answer.items(), key=lambda x: x[1], reverse=True)
            best_candi = shareholder_in_answer_list[0][0]
        else:
            shareholder_list = sorted(shareholder.items(), key=lambda x: x[1], reverse=True)
            best_candi = shareholder_list[0][0]
        for i, one_answer in enumerate(all_answer):
            if '增持的股东' not in one_answer.keys():
                all_answer[i]['增持的股东'] = best_candi
        # 通过对比最长公共子序列的方法找全称
        for one_answer in all_answer:
            for s in shareholder.keys():
                if s == one_answer['增持的股东']:
                    continue
                lcs = find_lcseque(s, one_answer['增持的股东'])
                if lcs == one_answer['增持的股东']:
                    # 如果只是单纯的多个空格，则不认为是全称
                    if ''.join(s.split()) == one_answer['增持的股东']:
                        continue
                    one_answer['增持的股东'] = s
                    break
    return all_answer


def equity_overweight(role_pred, original, sentences, doc_id, events=None):
    role_list = ['增持的股东', '增持金额', '增持开始日期']
    answer_table = {'event_type': '股东增持'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))

    all_answer = []
    if len(pred_list) == 0:
        one_answer = copy.deepcopy(answer_table)
        one_answer = slim_answer(one_answer)
        all_answer.append(one_answer)
    else:
        pred_list = clean_pred_list(pred_list)
        flat_pred_list = flatten_pred_list(pred_list)
        text = ' '.join(sentences)
        regex = r'(本次)?增持((公司)?股[票份])?(数\s?量|股\s?数|数)\s?\((万?股|)\)'
        table_search = re.search(regex, text)
        if table_search is None:
            all_answer = process_text(pred_list, flat_pred_list, original, sentences, answer_table, doc_id, events)
        else:
            all_answer = process_table(pred_list, flat_pred_list, original, sentences,
                                       answer_table, doc_id, table_search, events)

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
            event_role_pred = simplify_and_filter_pred(event_role_pred, event_type_pred)

            events = line['events']
            original = line['original'] if mode == 3 else sentences
            all_answer = equity_overweight(
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

    event_type = '股东增持'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
