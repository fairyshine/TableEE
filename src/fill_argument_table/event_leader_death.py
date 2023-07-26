#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import os
import re
import copy
import numpy as np


def search_age_by_rule(sentences):
    def find_age(sent):
        pattern = r'((?:[1-9][0-9]|[二三四五六七八九]?十?[一二三四五六七八九])周?岁)'
        res = re.finditer(pattern, sent)
        span = []
        for r in res:
            espan = r.span()
            entity_span = [espan[0], espan[1]-1, '死亡年龄']
            span.append(entity_span)
        return span
    ret = []
    for sent_id, sent in enumerate(sentences):
        age_span = find_age(sent)
        if len(age_span) == 0:
            continue
        ret.append([sent_id, age_span])
    return ret


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


def flatten_pred_list(pred_list):
    """
    将已经分组的预测列表展开（多维->一维）
    """
    ret = []
    for sub_pred in pred_list:
        ret += sub_pred
    return ret


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
    """
    找出与当前要素内联距离最近的一个组合
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


def global_filling_with_trigger(cur_idx, one_answer, global_dis, flat_pred_list, doc_id, max_dis=None):
    """
    找出与当前要素内联距离最近的一个组合
    """
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
        if cur_role == 'trigger':
            continue
        if idx > cur_idx and cur_role != '死亡年龄':
            continue
        if one_answer[cur_role] is None:
            one_answer[cur_role] = cur_entity
            if cur_role != '公司名称':
                indices.append(idx)
    total_dis = 0
    indices = sorted(indices)
    for i in range(len(indices)-1):
        total_dis += global_dis[indices[i]][indices[i+1]]
    one_answer = slim_answer(one_answer)
    return one_answer, total_dis


def leader_death(role_pred, original, sentences, doc_id, events=None):
    """
    通过一定的触发词找寻事件，而且由于行文习惯，事件要素基本都在触发词前面
    遗留问题：
    死亡/失联时间存在误判
    """
    role_list = ['公司名称', '高层人员', '高层职务', '死亡/失联时间', '死亡年龄']
    answer_table = {'event_type': '高层死亡'}
    answer_table.update({x: None for x in role_list})

    age_pred = search_age_by_rule(original)
    role_pred += age_pred
    pred_list = get_pred_list(role_pred, original, set(role_list))

    all_answer = []
    one_answer = copy.deepcopy(answer_table)
    if len(pred_list) > 0:
        flat_pred_list = flatten_pred_list(pred_list)
        noise_jobs = ['核心技术人员', '核心员工', '核心技术员工']
        flat_pred_list = list(filter(
            lambda x: x[-1] not in noise_jobs, flat_pred_list)
        )
        trigger_span = []
        for i, sent in enumerate(sentences):
            trigger_iter = re.finditer(r'去世|逝世|辞世|病逝|离世|病世|離世|身亡', sent)
            for trigger in trigger_iter:
                start, end = trigger.span()
                trigger_span.append([i, start, end-1, 'trigger', trigger.group()])
        assert len(trigger_span) > 0
        trigger_flat_pred_list = sort_pred_list(trigger_span + flat_pred_list)
        global_dis = cal_global_distance(trigger_flat_pred_list, sentences)
        best_ans = None
        min_dis = float('inf')
        for i, pred in enumerate(trigger_flat_pred_list):
            role = pred[-2]
            if role != 'trigger':
                continue
            one_answer = copy.deepcopy(answer_table)
            one_answer, total_dis = global_filling_with_trigger(
                i, one_answer, global_dis, trigger_flat_pred_list, doc_id)

            if best_ans is None:
                best_ans = one_answer
                min_dis = total_dis
            else:
                if len(one_answer) > len(best_ans):
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
        one_answer = best_ans
        if '公司名称' not in one_answer.keys():
            company_dict = {}
            for pred in trigger_flat_pred_list:
                if pred[-2] == '公司名称':
                    company_dict[pred[-1]] = company_dict.get(pred[-1], 0) + 1
            company_list = sorted(company_dict.items(), key=lambda x: x[1], reverse=True)
            if len(company_list) > 0:
                one_answer['公司名称'] = company_list[0][0]

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
            all_answer = leader_death(
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

    event_type = '高层死亡'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
