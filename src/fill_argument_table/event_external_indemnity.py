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


def remove_role(pred_list, role, right_entity):
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
    若最后一句中找不到，则在前文中搜索句子中出现"公告","公布"（关键词）的句子
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
        keywords = ['公告', '公布']
        for sub_pred in reversed(pred_list):
            for pred in sub_pred:
                sent_id = pred[0]
                role = pred[3]
                entity = pred[4]
                if role == '公告时间':
                    flag = any(word in sentences[sent_id] for word in keywords)
                    if len(entity) > 1 and flag:
                        one_answer[role] = entity
                        break
            if one_answer['公告时间'] is not None:
                break
    pred_list = remove_role(pred_list, '公告时间', one_answer['公告时间'])
    return pred_list


def find_company(pred_list, one_answer, sentences):
    """
    找公司名称，先找最后一句（一般和公告时间一同出现）
    否则就找最后一个公告时间所在位置附近的公司名称，若在最后一个公告时间之后的句子里
    还有公司名称出现，则放弃
    对于找到的公司名称，判断前面是否有"子公司"、"股东"等关键词（有子公司就选子公司）
    判断后面是否有"下属公司"字样，如果有的话，说明实际公司是其子公司
    尽可能选择全称
    """
    last_pred = pred_list[-1]
    company_candidate = None
    if len(sentences) == last_pred[0][0] + 1:
        role_set = set([x[3] for x in last_pred])
        if '公告时间' in role_set and '公告时间' in role_set:
            company_list = list(filter(lambda x: x[3] == '公司名称', last_pred))
            if len(company_list) != 0:
                company_set = set(x[-1] for x in company_list)
                if len(company_set) == 1:
                    company_candidate = company_list[0][-1]
                else:
                    time_start = time_end = 0
                    for pred in reversed(last_pred):
                        if pred[3] == '公告时间':
                            time_start = pred[1]
                            time_end = pred[2]
                            break
                    cur_min_dis = len(sentences[-1])
                    for pred in last_pred:
                        if pred[3] != '公司名称':
                            continue
                        dis = time_start - \
                            pred[2] if pred[1] < time_start else pred[1]-time_end
                        assert dis >= 0
                        if dis < cur_min_dis:
                            company_candidate = pred[-1]
                            cur_min_dis = dis
    else:
        for sub_pred in reversed(pred_list):
            role_set = set([x[3] for x in sub_pred])
            if '公告时间' not in role_set and '公司名称' in role_set:
                break
            if '公告时间' in role_set and '公司名称' in role_set:
                company_list = list(
                    filter(lambda x: x[3] == '公司名称', last_pred))
                if len(company_list) != 0:
                    company_set = set(x[-1] for x in company_list)
                    if len(company_set) == 1:
                        company_candidate = company_list[0][-1]
                    else:
                        time_start = time_end = 0
                        for pred in reversed(sub_pred):
                            if pred[3] == '公告时间':
                                time_start = pred[1]
                                time_end = pred[2]
                                break
                        cur_min_dis = len(sentences[sub_pred[0][0]])
                        for pred in sub_pred:
                            if pred[3] != '公司名称':
                                continue
                            dis = time_start - \
                                pred[2] if pred[1] < time_start else pred[1]-time_end
                            assert dis >= 0
                            if dis < cur_min_dis:
                                company_candidate = pred[-1]
                                cur_min_dis = dis
                    break
    found = False if company_candidate is None else True
    if found:
        for sub_pred in pred_list:
            for pred in sub_pred:
                # sent_id = pred[0]
                role = pred[3]
                entity = pred[4]
                if role != '公司名称':
                    continue
                if company_candidate in entity:
                    company_candidate = entity
        one_answer['公司名称'] = company_candidate
        pred_list = remove_role(pred_list, '公司名称', one_answer['公司名称'])
    return pred_list, found


def find_money(pred_list, one_answer, sentences):
    pass


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
            try:
                role = sub_role[3]
            except:
                import pdb
                pdb.set_trace()

            matrix[sent_id][role2idx[role]] = 1
    times = np.sum(matrix, axis=0)
    idf = np.log((sent_num+1) / (times+1))
    scores = np.sum(idf * matrix, axis=1)
    scores = scores.tolist()
    max_score = max(scores)
    scores = [score+max_score if valid_sent[i]
              else score for i, score in enumerate(scores)]
    scores_idx = [(scores[i], i) for i in range(sent_num)]
    scores_idx = sorted(scores_idx, key=lambda x: x[0], reverse=True)
    return scores_idx


def find_key_sent(pred_list, role_list, sentences):
    role2idx = {role: i for i, role in enumerate(role_list)}
    sent_num = len(sentences)
    role_num = len(role2idx)
    matrix = np.zeros(shape=(sent_num, role_num))
    for sub_pred in pred_list:
        for pred in sub_pred:
            sent_id = pred[0]
            role = pred[3]
            matrix[sent_id][role2idx[role]] = 1
    occur_times = np.sum(matrix, axis=0)
    idf = np.log((sent_num+1) / (occur_times+1))
    scores = np.sum(idf * matrix, axis=1)
    scores = scores.tolist()
    scores_idx = [(scores[i], i) for i in range(sent_num)]
    scores_idx = sorted(scores_idx, key=lambda x: x[0], reverse=True)
    return scores_idx


def external_indemnity(role_pred, original, sentences, doc_id, events):
    role_list = ['公司名称', '公告时间', '赔付对象', '赔付金额']
    answer_table = {'event_type': '重大对外赔付'}
    answer_table.update({x: None for x in role_list})
    pred_list = get_pred_list(role_pred, original, set(role_list))

    all_answer = []
    one_answer = copy.deepcopy(answer_table)
    if len(pred_list) > 0:
        pred_list = find_announcement_time(pred_list, one_answer, sentences)
        pred_list, found = find_company(pred_list, one_answer, sentences)
        scores_idx = find_key_sent(pred_list, role_list, sentences)
        sent2pred = {sub_pred[0][0]: sub_pred for sub_pred in pred_list}
        role2entity = {}
        for sub_pred in pred_list:
            for pred in sub_pred:
                role = pred[3]
                entity = pred[4]
                if role not in role2entity.keys():
                    role2entity[role] = {}
                role2entity[role][entity] = role2entity[role].get(entity, 0)+1
        for _, idx in scores_idx:
            if idx not in sent2pred.keys():
                continue
            sub_role_list = sent2pred[idx]
            for sub_role in sub_role_list:
                role = sub_role[3]
                entity = sub_role[4]
                if role == '公告时间':
                    continue
                if found and role == '公司名称':
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
            event_role_pred = simplify_and_filter_pred(
                event_role_pred, event_type_pred)

            events = line['events']
            original = line['original'] if mode == 3 else sentences
            all_answer = external_indemnity(
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

    event_type = '重大对外赔付'
    path_src = '../noise_data/noise_{}.json'.format(event_type)
    path_des = '../fusion_result/result_of_{}.txt'.format(event_type)
    generate_answer(path_src, path_des)
