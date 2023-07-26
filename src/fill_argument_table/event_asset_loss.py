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


def cal_sent_scores(role_list, answer_table, sentences, valid_sent):
    role2idx = {}
    for key in answer_table.keys():
        if key == 'event_type':
            continue
        role2idx[key] = len(role2idx)
    sent_num = len(sentences)
    matrix = np.zeros(shape=(sent_num, len(role2idx)))
    for sub_role_list in role_list:
        for sub_role in sub_role_list:
            sent_id = sub_role[0]
            role = sub_role[3]
            matrix[sent_id][role2idx[role]] = 1
    times = np.sum(matrix, axis=0)
    idf = np.log((sent_num+1) / (times+1))
    scores = np.sum(idf * matrix, axis=1)
    scores = scores.tolist()
    max_score = max(scores)
    scores = [score+max_score if valid_sent[i] else score for i, score in enumerate(scores)]
    scores_idx = [(scores[i], i) for i in range(sent_num)]
    scores_idx = sorted(scores_idx, key=lambda x: x[0], reverse=True)
    return scores_idx


def clean_role_list(role_list, time=None):
    new_role_list = []
    for sub_role_list in role_list:
        tmp = []
        for sub_role in sub_role_list:
            role = sub_role[3]
            entity = sub_role[4]
            if time is not None:
                if role == '公告时间' and entity != time:
                    continue
            if len(entity) == 1:
                continue
            tmp.append(sub_role)
        if len(tmp) != 0:
            new_role_list.append(tmp)
    return new_role_list


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


def asset_loss(role_pred, original, sentences, doc_id, events):
    role_list = ['公司名称', '公告时间', '损失金额', '其他损失']
    answer_table = {'event_type': '重大资产损失'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))

    all_answer = []
    one_answer = copy.deepcopy(answer_table)
    if len(pred_list) > 0:
        for sub_role_list in reversed(pred_list):
            for sub_role in reversed(sub_role_list):
                role = sub_role[3]
                entity = sub_role[4]
                if role == '公告时间':
                    if len(entity) > 1:
                        one_answer[role] = entity
                        break
            if one_answer['公告时间'] is not None:
                break

        role_list = clean_role_list(pred_list, time=one_answer['公告时间'])
        keywords = ['损失', '资产']
        valid_sent = []
        for i, sent in enumerate(original):
            flag = any(word in sent for word in keywords)
            valid_sent.append(flag)
        scores_idx = cal_sent_scores(role_list, answer_table, original, valid_sent)

        idx2role_list = {}
        for sub_role_list in role_list:
            sent_id = sub_role_list[0][0]
            idx2role_list[sent_id] = sub_role_list
        role2entity = {}
        for sub_role_list in role_list:
            for sub_role in sub_role_list:
                role = sub_role[3]
                entity = sub_role[4]
                if role not in role2entity.keys():
                    role2entity[role] = {}
                role2entity[role][entity] = role2entity[role].get(entity, 0)+1
        for _, idx in scores_idx:
            if idx not in idx2role_list.keys():
                continue
            sub_role_list = idx2role_list[idx]
            for sub_role in sub_role_list:
                role = sub_role[3]
                entity = sub_role[4]
                if role == '公告时间':
                    continue
                if one_answer[role] is None:
                    one_answer[role] = entity
                else:
                    role_entity_dict = role2entity[role]
                    if one_answer[role] == entity:
                        continue
                    elif one_answer[role] in entity or entity in one_answer[role]:
                        if one_answer[role] in entity:
                            one_answer[role] = entity
                    else:
                        if role_entity_dict[entity] > role_entity_dict[one_answer[role]]:
                            one_answer[role] = entity
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
            all_answer = asset_loss(
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

    event_type = '重大资产损失'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
