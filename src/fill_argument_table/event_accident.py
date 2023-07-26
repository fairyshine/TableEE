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


def flatten_pred_list(pred_list):
    """
    将已经分组的预测列表展开（多维->一维）
    """
    ret = []
    for sub_pred in pred_list:
        ret += sub_pred
    return ret


def slim_answer(one_answer):
    """
    去除答案中值为None的要素
    """
    ret = {}
    for role, entity in one_answer.items():
        if entity is not None:
            ret[role] = entity
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


def remove_role(pred_list, role, right_entity=None):
    # 将错误的公告事件从pred_list中移除
    ret = []
    for sub_pred in pred_list:
        tmp = []
        for pred in sub_pred:
            cur_role = pred[3]
            cur_entity = pred[4]
            if len(cur_entity) < 2:  # 长度＜2的实体直接删除
                continue
            if cur_role == role and cur_entity != right_entity:
                continue
            tmp.append(pred)
        if len(tmp) > 0:
            ret.append(tmp)
    return ret


def find_announcement_time(pred_list, one_answer, sentences):
    """
    找公告时间（从后往前找）,限于最后一句
    若最后一句中找不到，则在前文中搜索句子中出现"特此公告","公告","公布"（关键词）的句子
    判断其中是否有公告时间，从后往前找
    如果找到公告时间的话，则把文中其他公告时间删除
    """
    last_pred = pred_list[-1]
    # 若过预测结果的最后一个是文本的最后一句时
    if len(sentences) == last_pred[0][0] + 1:
        for pred in reversed(last_pred):
            role = pred[3]
            entity = pred[4]
            if role == '公告时间':
                if len(entity) > 1:
                    one_answer[role] = entity
                    break
    if one_answer['公告时间'] is None:
        keywords = ['特此公告', '公告', '公布']
        for sub_pred in reversed(pred_list):
            for pred in reversed(sub_pred):
                sent_id, start, end, role, entity = pred
                if role == '公告时间':
                    context = sentences[sent_id][max(start-30, 0):end+20]
                    flag = any(word in context for word in keywords)
                    if len(entity) > 1 and flag:
                        one_answer[role] = entity
                        break
            if one_answer['公告时间'] is not None:
                break
    pred_list = remove_role(pred_list, '公告时间', one_answer['公告时间'])
    return pred_list


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


def global_filling(cur_idx, one_answer, global_dis, flat_pred_list, max_dis=None):
    pred = flat_pred_list[cur_idx]
    pred_role = pred[3]
    one_answer[pred_role] = pred[-1]
    cur_dis = [(idx, x) for idx, x in enumerate(global_dis[cur_idx])]
    sorted_dis = sorted(cur_dis, key=lambda x: x[1], reverse=False)
    indices = [cur_idx]
    for idx, dis in sorted_dis:
        if dis == 0:
            continue
        if max_dis is not None and dis > max_dis:
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


def accident(role_pred, original, sentences, doc_id, events=None):
    """
    遗留问题：
    公司名称：子公司还是母公司，全称和简称
    伤亡人数：无法确定
    其他影响：有一定规律，待完善
    """
    role_list = ['公司名称', '公告时间', '伤亡人数', '损失金额', '其他影响']
    answer_table = {'event_type': '重大安全事故'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))

    all_answer = []
    one_answer = copy.deepcopy(answer_table)
    if len(pred_list) > 0:
        pred_list = find_announcement_time(pred_list, one_answer, sentences)
        flat_pred_list = flatten_pred_list(pred_list)
        global_dis = cal_global_distance(flat_pred_list, sentences)
        best_ans = None
        min_dis = float('inf')
        tmp_ans = one_answer
        for i, pred in enumerate(flat_pred_list):
            one_answer = copy.deepcopy(tmp_ans)
            one_answer, total_dis = global_filling(i, one_answer, global_dis, flat_pred_list)
            if best_ans is None:
                best_ans = one_answer
                min_dis = total_dis
            else:
                if total_dis < min_dis:
                    best_ans = one_answer
                    min_dis = total_dis
        one_answer = best_ans
        company = {}
        for i, pred in enumerate(flat_pred_list):
            role = pred[-2]
            entity = pred[-1]
            if role == '公司名称':
                company[entity] = company.get(entity, 0) + 1

        if '其他影响' not in one_answer.keys():
            industry = {}
            for pred in flat_pred_list:
                if pred[-2] == '其他影响':
                    industry[pred[-1]] = industry.get(pred[-1], 0) + 1
            if len(industry) > 0:
                industry_list = sorted(industry.items(), key=lambda x: x[1], reverse=True)
                one_answer['其他影响'] = industry_list[0][0]

    one_answer = slim_answer(one_answer)

    all_answer.append(one_answer)
    return all_answer


def show_role_pred(role_list):
    for sub_role_list in role_list:
        print(sub_role_list)


def show_sent(sentences):
    for i, sent in enumerate(sentences):
        print('{:>2d}\t{}'.format(i, sent))


def simplify_and_filter_pred(event_role_pred, event_type_pred):
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
            all_answer = accident(
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

    event_type = '重大安全事故'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
