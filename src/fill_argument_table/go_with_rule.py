#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import json
import os

from .event_equity_underweight import equity_underweight
from .event_bankruptcy import bankruptcy
from .event_equity_overweight import equity_overweight
from .event_equity_freeze import equity_freeze
from .event_equity_pledge import equity_pledge
from .event_asset_loss import asset_loss
from .event_accident import accident
from .event_leader_death import leader_death
from .event_external_indemnity import external_indemnity


def event_table_filling(event_type, role_pred, original, sentences, doc_id, events=None):
    function = {
        '股东减持': equity_underweight,
        '破产清算': bankruptcy,
        '股东增持': equity_overweight,
        '股权冻结': equity_freeze,
        '股权质押': equity_pledge,
        '重大资产损失': asset_loss,
        '重大安全事故': accident,
        '高层死亡': leader_death,
        '重大对外赔付': external_indemnity,
    }
    all_answer = function[event_type](
        role_pred, original, sentences, doc_id, events)
    return all_answer


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


def fill_event_table(meta):
    event_type_pred = meta['event_type_pred']
    event_role_pred = meta['event_role_pred']
    original = meta['original']
    sentences = meta['sentences']
    doc_id = meta['doc_id']

    event_role_pred = simplify_and_filter_pred(event_role_pred, event_type_pred)
    all_answer = event_table_filling(
        event_type_pred,
        event_role_pred,
        original,
        sentences,
        doc_id,
    )
    ret = dict(
        doc_id=doc_id,
        events=all_answer,
    )
    return ret


if __name__ == '__main__':
    data_input = {}

    data_input['event_type_pred'] = '破产清算'
    data_input['event_role_pred'] = [(0,[(31,44,'破产清算-公司名称')]),(1,[(6,14,'破产清算-受理法院'),(116,129,'破产清算-公司名称'),(137,145,'破产清算-公告时间')])]
    data_input['original'] = ["证券代码:400027证券简称:生态1编号:临2012-003湖北江湖生态农业股份有限公司破产重整管理人关于债权裁定确认等情况的公告本管理人及其成员保证本公告内容的真实、准确和完整,没有虚假记载、误导性陈述或者重大遗漏。", "本管理人收到荆州市中级人民法院作出的[2010]鄂荆中民破字第5-8、5-9、5-12号《民事裁定》,裁定确认债权3,546,884,148.47元;批准《财产管理方案》;对《财产变价方案》中的变价方式,采取拍卖方式执行。特此公告。湖北江湖生态农业股份有限公司破产重整管理人二〇一二年三月二日"]
    data_input['sentences'] = data_input['original']
    data_input['doc_id'] = '2649905'
    meta = fill_event_table(data_input)
    print(meta)

'''
{"content": "证券代码：400027证券简称：生态1编号：临2012-003湖北江湖生态农业股份有限公司破产重整管理人关于债权裁定确认等情况的公告本管理人及其成员保证本公告内容的真实、准确和完整，没有虚假记载、误导性陈述或者重大遗漏。本管理人收到荆州市中级人民法院作出的[2010]鄂荆中民破字第5-8、5-9、5-12号《民事裁定》，裁定确认债权3,546,884,148.47元；批准《财产管理方案》；对《财产变价方案》中的变价方式，采取拍卖方式执行。特此公告。湖北江湖生态农业股份有限公司破产重整管理人二〇一二年三月二日", 
"doc_id": "2649905", 
"events": [{"event_id": "4546472", "受理法院": "荆州市中级人民法院", "公司名称": "湖北江湖生态农业股份有限公司", "公告时间": "二〇一二年三月二日", "event_type": "破产清算"}], 
"processed_content": "证券代码：400027证券简称：生态1编号：临2012-003湖北江湖生态农业股份有限公司破产重整管理人关于债权裁定确认等情况的公告本管理人及其成员保证本公告内容的真实、准确和完整，没有虚假记载、误导性陈述或者重大遗漏。本管理人收到荆州市中级人民法院作出的[2010]鄂荆中民破字第5-8、5-9、5-12号《民事裁定》，裁定确认债权3,546,884,148.47元；批准《财产管理方案》；对《财产变价方案》中的变价方式，采取拍卖方式执行。特此公告。湖北江湖生态农业股份有限公司破产重整管理人二〇一二年三月二日", 
"sentences": ["证券代码:400027证券简称:生态1编号:临2012-003湖北江湖生态农业股份有限公司破产重整管理人关于债权裁定确认等情况的公告本管理人及其成员保证本公告内容的真实、准确和完整,没有虚假记载、误导性陈述或者重大遗漏。", "本管理人收到荆州市中级人民法院作出的[2010]鄂荆中民破字第5-8、5-9、5-12号《民事裁定》,裁定确认债权3,546,884,148.47元;批准《财产管理方案》;对《财产变价方案》中的变价方式,采取拍卖方式执行。特此公告。湖北江湖生态农业股份有限公司破产重整管理人二〇一二年三月二日"], 
"entity_span": {"湖北江湖生态农业股份有限公司": [[0, 31, 44], [1, 116, 129]], "二〇一二年三月二日": [[1, 137, 145]], "荆州市中级人民法院": [[1, 6, 14]]}, 
"role_span": {"破产清算-公司名称-湖北江湖生态农业股份有限公司": [[0, 31, 44], [1, 116, 129]], "破产清算-公告时间-二〇一二年三月二日": [[1, 137, 145]], "破产清算-受理法院-荆州市中级人民法院": [[1, 6, 14]]}}
'''