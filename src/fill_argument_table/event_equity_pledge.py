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


def clean_pred_list(pred_list):
    """
    去除实体长度小于等于1的事件要素
    经统计，在目前的训练数据中，没有长度小于1的要素
    # 在训练数据中，质押金额是没有`,，`这样的间隔符的
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
    if len(ret) == 0:
        ret.append(best_ans)
    return ret
    # return all_answer


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


def complete_one_answer_v2(cur_idx, one_answer, global_dis, flat_pred_list, max_dis=100):
    """
    如果文中出现多个质押金额（≥2），使用此方案
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
        cur_sent_id = cur_pred[0]
        if abs(cur_sent_id-pred[0]) > 1:
            break
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


def complete_one_answer_v3(cur_idx, one_answer, global_dis, flat_pred_list, pred_in_table_flag, max_dis=100):
    """
    如果文中含有表格，除表格外的文本也识别出了表格，则使用此方案
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
        if pred_in_table_flag[idx] is True:
            break
        cur_pred = flat_pred_list[idx]
        cur_sent_id = cur_pred[0]
        if abs(cur_sent_id-pred[0]) > 1:
            break
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
    all_answer.append(best_ans)
    return all_answer


def generate_event_2(flat_pred_list, sentences, answer_table):
    """
    如果出现质押金额（一个金额值），就找出内联距离（要素之间距离之和）最小的一个组合
    且保证要素的数量尽可能地多
    该组合以质押金额为中心进行定位
    只返回一个最佳结果
    """
    all_answer = []
    global_dis = cal_global_distance(flat_pred_list, sentences)
    best_ans = None
    min_dis = float('inf')
    for i, pred in enumerate(flat_pred_list):
        if pred[3] != '质押金额':
            continue
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
    all_answer.append(best_ans)
    return all_answer


def generate_event_3(flat_pred_list, sentences, answer_table):
    """
    如果文中出现多个质押金额（≥2），就以质押金额为中心，找出若干个组合
    取内联距离最小的两个组合
    """
    tmp_all_answer = []
    global_dis = cal_global_distance(flat_pred_list, sentences)
    for i, pred in enumerate(flat_pred_list):
        if pred[3] != '质押金额':
            continue
        one_answer = copy.deepcopy(answer_table)
        one_answer, total_dis = complete_one_answer_v2(i, one_answer, global_dis, flat_pred_list)
        tmp_all_answer.append((one_answer, total_dis))
    tmp_all_answer = sorted(tmp_all_answer, key=lambda x: x[1], reverse=False)
    all_answer = [x[0] for x in tmp_all_answer[:2]]
    return all_answer


def process_text(flat_pred_list, sentences, answer_table):
    """
    用于处理篇章中没有出现表格的文本
    """
    pledge_amount = len(
        set([p[-1] for p in
             filter(lambda x: x[3] == '质押金额', flat_pred_list)]))
    if pledge_amount == 0:
        all_answer = generate_event_1(flat_pred_list, sentences, answer_table)
    elif pledge_amount == 1:
        all_answer = generate_event_2(flat_pred_list, sentences, answer_table)
    else:
        all_answer = generate_event_2(flat_pred_list, sentences, answer_table)
    return all_answer


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


def process_table(flat_pred_list, sentences, original, answer_table, doc_id=None):
    """
    尽可能地提取出表格中的内容
    """
    # 将句子拼成文章，句子级别的预测也转为全文的预测（就下标而言）
    text, original_text, global_pred_list, sent_span = sent2text(sentences, original, flat_pred_list)
    # 找到全文所有表头的位置
    table_regex = r'(?:所\s?持\s?有?\s?的?\s?)?股?[份本]\s?的?\s?比例(?:\s?\(%\))?(?:\s?用\s?途)?|股\s?本\s?总\s?额\s?的?\s?比例(?:\s?\(%\))?(?:\s?用\s?途)?'
    table_iter = re.finditer(table_regex, text)
    table_match = []
    for table_header in table_iter:
        table_match.append(table_header)
    # pred_in_table_flag中为True表示元素在表格内容中
    pred_in_table_flag = [False] * len(global_pred_list)
    # 若果没有表头，则返回空
    if len(table_match) == 0:
        return [], pred_in_table_flag
    # 通过识别的表头将文章切成若干部分
    # 前一个表头到下一个表头的开始位置前为一个片段
    # 或者到下一段落结束位置（为了处理因为文章断句把表格信息割裂的情况）
    table_span = []
    for i in range(len(table_match)-1):
        he = table_match[i].span()[1]
        cur_idx = None
        for sent_id, span in enumerate(sent_span):
            if he-1 >= span[0] and he-1 <= span[1]:
                cur_idx = sent_id
                break
        assert cur_idx is not None
        ts = he
        te = sent_span[cur_idx][1] if cur_idx == len(sentences)-1\
            else sent_span[cur_idx+1][1]
        next_hs = table_match[i+1].span()[0]
        if next_hs-1 < te:
            te = next_hs - 1
        table_span.append((ts, te))
    # 如果表头就匹配到一个或者处理最后一个表头
    last_he = table_match[-1].span()[1]
    last_idx = None
    for sent_id, span in enumerate(sent_span):
        if last_he-1 >= span[0] and last_he-1 <= span[1]:
            last_idx = sent_id
            break
    assert last_idx is not None
    last_ts = last_he
    last_te = sent_span[last_idx][1] if last_idx == len(sentences)-1\
        else sent_span[last_idx+1][1]
    table_span.append((last_ts, last_te))

    # 找出文本中所有的百分数（大概率是股份比例）
    percent_iter = re.finditer(r'(?:\d|[1-9]\d|100)(?:\.\d+)?%', text)
    percent_match = []
    for percent in percent_iter:
        percent_match.append(percent)

    # 有部分表格里的百分数没有百分号，如果百分数匹配不到的话，小数匹配才有效
    decimal_iter = re.finditer(r'(?:\d|[1-9]\d|100)(?:\.\d+)', text)
    decimal_match = []
    for decimal in decimal_iter:
        decimal_match.append(decimal)

    # 表格中的日期比较标准，用正则可匹配找出
    date_iter = re.finditer(
        r'20[0-2]\d年[01]?\d月[0-3]?\d日?|20[0-2]\d\.[01]?\d\.[0-3]?\d|20[0-2]\d-[01]?\d-[0-3]?\d|20[0-2]\d/[01]?\d/[0-3]?\d', text)
    date_match = []
    for date in date_iter:
        date_match.append(date)

    # 表格中的质押方应该是不缺失的，这里统计整篇文章中不同质押方出现的次数
    # 后面处理时，如果答案缺失质押方，则挑选高频的质押方予以补充
    pledgor = {}
    for pred in flat_pred_list:
        if pred[3] == '质押方':
            pledgor[pred[-1]] = pledgor.get(pred[-1], 0) + 1
    pledgor_list = sorted(pledgor.items(), key=lambda x: x[1], reverse=True)
    best_pledgor_candi = pledgor_list[0][0] if len(pledgor) > 0 else None

    all_answer = []
    for table_id, (ts, te) in enumerate(table_span):
        # 找出当前表格可能范围内的百分数和日期
        cur_percent = []
        for percent in percent_match:
            p_start, p_end = percent.span()
            if p_start >= ts and p_end-1 <= te:
                cur_percent.append(percent)
        if len(cur_percent) == 0:
            # 如果没有找到百分数，就将有效范围内的小数作为百分数
            for decimal in decimal_match:
                p_start, p_end = decimal.span()
                if p_start >= ts and p_end-1 <= te:
                    cur_percent.append(decimal)
        # 通过表头和百分数位置将文本切割（百分数应出现在表头之后）
        indices = [(ts, ts)]
        for p in cur_percent:
            p_start = p.span()[0]
            if p_start >= ts:
                indices.append(p.span())
        one_table_answer = []
        if len(indices) != 0:
            for i in range(len(indices)-1):
                start = indices[i][1]
                end = indices[i+1][0]-1
                valid_pred_list = []
                # 定位表格中的某一行，找出目前内容的预测结果
                for pred in global_pred_list:
                    pred_start = pred[0]
                    pred_end = pred[1]
                    if pred_start >= start and pred_end <= end:
                        valid_pred_list.append(pred)
                # 找出当前行中的日期（质押开始日期、质押结束日期）
                valid_date = []
                for date in date_match:
                    date_start, date_end = date.span()
                    if date_start >= start and date_end-1 <= end:
                        valid_date.append((date_start, date_end-1, original_text[date_start:date_end]))

                # 没有日期或日期过多的可以排除不是表格内容
                if len(valid_date) == 0 or len(valid_date) > 3:
                    break
                one_answer = copy.deepcopy(answer_table)
                ans_span = {}
                # 质押开始日期是当前行识别出的第一个日期
                # 结束日期可能有，可能无，选择不同于开始日期的最后一个日期
                one_answer['质押开始日期'] = valid_date[0][-1]
                ans_span['质押开始日期'] = (valid_date[0][0], valid_date[0][1])
                if len(valid_date) > 1:
                    one_answer['质押结束日期'] = valid_date[-1][-1]
                    ans_span['质押结束日期'] = (valid_date[-1][0], valid_date[-1][1])
                # 用预测出的结果填充答案
                # 在目前的表格格式下，质押方需要在质押开始日期前面
                for pred in valid_pred_list:
                    role = pred[2]
                    if one_answer[role] is None:
                        one_answer[role] = pred[-1]
                        ans_span[role] = (pred[0], pred[1])
                    # 避免把模型误把质押方识别出接收方
                    elif role == '质押方':
                        if pred[1] < valid_date[0][0]:
                            one_answer[role] = pred[-1]
                            ans_span[role] = (pred[0], pred[1])
                # 质押金额在质押开始日期之前，用正则找出
                money_start, money_end = None, None
                if one_answer['质押金额'] is None:
                    sub_string = text[start: valid_date[0][0]]
                    sub_original = original_text[start: valid_date[0][0]]
                    pledge_amount = re.search(r'[1-9]\d*(?:\.\d+)?万?', sub_string)
                    if pledge_amount is not None:
                        money_start, money_end = pledge_amount.span()
                        one_answer['质押金额'] = sub_original[money_start:money_end]
                        money_start += start
                        money_end += start-1
                        ans_span['质押金额'] = (money_start, money_end)
                # 质押方在质押金额之前，每一行开始的前部分
                # 如果表头中没有“用途”或者这是第一行，则选则空格的第一部分
                # 否则就选第二部分（主要是上一行的“用途”被分割在下一行）
                if one_answer['质押方'] is None:
                    if one_answer['质押金额'] is not None:
                        entity_start = start
                        entity_end = ans_span['质押金额'][0]-1
                        sub_string = original_text[entity_start: entity_end]
                        ts, te = table_match[table_id].span()
                        table_header = text[ts: te]
                        sub_string_list = sub_string.strip().split()
                        # 一般来说，每一列信息之间是有空格隔开的
                        if len(sub_string_list) >= 2:
                            pledgor = None
                            if '用途' not in table_header or i == 0:
                                # 质押方需要在“是否为第一大股东”的答案之前
                                if sub_string_list[0] not in ['是', '否']:
                                    pledgor = sub_string_list[0]
                            else:
                                if sub_string_list[1] not in ['是', '否']:
                                    pledgor = sub_string_list[1]
                            if pledgor is not None:
                                beg_idx = sub_string.index(pledgor)
                                one_answer['质押方'] = pledgor
                                entity_start = entity_start + beg_idx
                                entity_end = entity_start + len(pledgor) - 1
                                ans_span['质押方'] = (entity_start, entity_end)
                # 接收方在质押结束日期之后
                if one_answer['接收方'] is None:
                    if one_answer['质押结束日期'] is not None and len(valid_date) > 1:
                        entity_start = valid_date[-1][1]+1
                        entity_end = end
                        while text[entity_start] in '!"#$%&\'*+,-./:;<=>?@[\\]^_`{|}~' or text[entity_start] == ' ':
                            entity_start = entity_start + 1
                        while text[entity_end] in '!"#$%&\'*+,-./:;<=>?@[\\]^_`{|}~' or text[entity_end] == ' ':
                            entity_end = entity_end - 1
                        if entity_start <= entity_end:
                            one_answer['接收方'] = original_text[entity_start:entity_end+1]
                            ans_span['接收方'] = (entity_start, entity_end)
                # 通过一些方法判断表格是否结束,一旦不是表格，停止检索
                table_stop = False
                # 如果生成的答案中要素＜2，不是表格内容。
                # 在训练数据中，所有股权质押事件都包含至少两个要素
                if len(ans_span) < 2:
                    table_stop = True
                else:
                    # 主要通过两个方面来判断：
                    # 要素的排序
                    # 目前答案中的要素span是否冲突
                    ans_span_list = sorted(ans_span.items(), key=lambda x: x[1][0])
                    role2id = {role: idx for idx, role
                               in enumerate(['质押方', '质押金额', '质押开始日期',
                                             '质押结束日期', '接收方'])}
                    for k in range(len(ans_span_list)-1):
                        cur = ans_span_list[k]
                        nxt = ans_span_list[k+1]
                        if cur[1][1] >= nxt[1][0]:
                            table_stop = True
                            break
                        if role2id[cur[0]] >= role2id[nxt[0]]:
                            table_stop = True
                            break
                if table_stop:
                    break
                else:
                    # 运行到此处，说明ans_span符合要求
                    for idx, pred in enumerate(global_pred_list):
                        pred_start = pred[0]
                        pred_end = pred[1]
                        if pred_start >= start and pred_end <= end:
                            pred_in_table_flag[idx] = True
                    # 如果当前质押方为空，则用前一行答案的质押方值填充
                    # 若前一个答案的质押方仍为空，则填入全文高频的质押方
                    if one_answer['质押方'] is None:
                        if len(one_table_answer) > 0 and '质押方' in one_table_answer[-1].keys():
                            one_answer['质押方'] = one_table_answer[-1]['质押方']
                        else:
                            one_answer['质押方'] = best_pledgor_candi
                    one_answer = slim_answer(one_answer)
                    one_table_answer.append(one_answer)
        all_answer += one_table_answer
    return all_answer, pred_in_table_flag


def equity_pledge(role_pred, original, sentences, doc_id, events=None):
    """
    主要思路：
    如果公告中有表格，答案就是表格内的答案；
    否则，就从全文找出内联距离最小的事件组合（若干）
    遗留问题：
    表格中的质押方难以确定边界
    引入日期比较，质押结束日期应晚于开始日期
    对比了下人工标注的部分数据：主要错在有些实体模型没识别出来，漏掉
    还有一些不同的表格、行之间合并的表格不好处理（数量很少）
    另外，对于同一质押金额转手质押的情况（互相穿插），难以区别，最多识别一个
    <br>还原处理待优化，可对比线上数据doc_id=2670226
    """
    role_list = ['质押方', '接收方', '质押金额', '质押开始日期', '质押结束日期']
    answer_table = {'event_type': '股权质押'}
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
        all_answer, pred_in_table_flag = process_table(
            flat_pred_list, sentences, original, answer_table, doc_id
        )

        if len(all_answer) > 0:
            pledgor = {}
            money = set()
            for i, pred in enumerate(flat_pred_list):
                if pred[3] == '质押金额' and not pred_in_table_flag[i]:
                    money.add(pred[-1])
                if pred[3] == '质押方':
                    pledgor[pred[-1]] = pledgor.get(pred[-1], 0) + 1
            # 如果表格中金额缺少“万”字，但其他地方有完整的数字，则选择完整的
            for idx, one_answer in enumerate(all_answer):
                if '质押金额' not in one_answer.keys():
                    continue
                if '万' in one_answer['质押金额']:
                    pass
                elif '{}万'.format(one_answer['质押金额']) in money:
                    all_answer[idx]['质押金额'] += '万'
        else:
            all_answer = process_text(flat_pred_list, sentences, answer_table)
            all_answer = post_process(all_answer)
        pledgor = set()
        pledgee = set()
        for pred in flat_pred_list:
            if pred[3] == '质押方':
                pledgor.add(pred[-1])
            elif pred[3] == '接收方':
                pledgee.add(pred[-1])
        tmp_answer = []
        for one_answer in all_answer:
            if '质押方' in one_answer.keys():
                for s in pledgor:
                    if s == one_answer['质押方']:
                        continue
                    lcs = find_lcseque(s, one_answer['质押方'])
                    if lcs == one_answer['质押方']:
                        # 如果只是单纯的多个空格，则不认为是全称
                        if ''.join(s.split()) == one_answer['质押方']:
                            continue
                        one_answer['质押方'] = s
                        break
            if '接收方' in one_answer.keys():
                for s in pledgee:
                    if s == one_answer['接收方']:
                        continue
                    lcs = find_lcseque(s, one_answer['接收方'])
                    if lcs == one_answer['接收方']:
                        # 如果只是单纯的多个空格，则不认为是全称
                        if ''.join(s.split()) == one_answer['接收方']:
                            continue
                        one_answer['接收方'] = s
                        break
            # <br>在预处理时被处理成了空格
            if '质押方' in one_answer.keys():
                if ' ' in one_answer['质押方']:
                    one_answer['质押方'] = re.sub(r'\s+', '<br>', one_answer['质押方'])
            if '接收方' in one_answer.keys():
                if ' ' in one_answer['接收方']:
                    one_answer['接收方'] = re.sub(r'\s+', '<br>', one_answer['接收方'])
            one_answer = slim_answer(one_answer)
            tmp_answer.append(one_answer)
        all_answer = tmp_answer

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
            all_answer = equity_pledge(
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

    event_type = '股权质押'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
