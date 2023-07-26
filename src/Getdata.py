#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import re
import json
from rich.progress import track
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer


EVENT_TYPE = ['破产清算','重大安全事故','股东减持',
            '股东增持','股权冻结','股权质押',
            '高层死亡','重大资产损失','重大对外赔付']
ARGUMENT_TYPE = {
    0:['公司名称', '公告时间', '公司行业', '裁定时间', '受理法院'],
    1:['公司名称', '公告时间', '伤亡人数', '损失金额', '其他影响'],
    2:['减持的股东', '减持开始日期', '减持金额'],
    3:['增持的股东', '增持开始日期', '增持金额'],
    4:['被冻结股东', '冻结开始日期', '冻结结束日期', '冻结金额'],
    5:['质押方', '接收方', '质押开始日期', '质押结束日期', '质押金额'],
    6:['公司名称', '高层人员', '高层职务', '死亡/失联时间', '死亡年龄'],
    7:['公司名称', '公告时间', '损失金额', '其他损失'],
    8:['公司名称', '公告时间', '赔付对象', '赔付金额']
}

ARGUMENT_TYPE_DICT = {
    0:{'公司名称':1, '公告时间':2, '公司行业':3, '裁定时间':4, '受理法院':5},
    1:{'公司名称':6, '公告时间':7, '伤亡人数':8, '损失金额':9, '其他影响':10},
    2:{'减持的股东':11, '减持开始日期':12, '减持金额':13},
    3:{'增持的股东':14, '增持开始日期':15, '增持金额':16},
    4:{'被冻结股东':17, '冻结开始日期':18, '冻结结束日期':19, '冻结金额':20},
    5:{'质押方':21, '接收方':22, '质押开始日期':23, '质押结束日期':24, '质押金额':25},
    6:{'公司名称':26, '高层人员':27, '高层职务':28, '死亡/失联时间':29, '死亡年龄':30},
    7:{'公司名称':31, '公告时间':32, '损失金额':33, '其他损失':34},
    8:{'公司名称':35, '公告时间':36, '赔付对象':37, '赔付金额':38}
}
ARGUMENT_TYPE_HASH = {
    0:{1:'公司名称', 2:'公告时间', 3:'公司行业', 4:'裁定时间', 5:'受理法院'},
    1:{6:'公司名称', 7:'公告时间', 8:'伤亡人数', 9:'损失金额', 10:'其他影响'},
    2:{11:'减持的股东', 12:'减持开始日期', 13:'减持金额'},
    3:{14:'增持的股东', 15:'增持开始日期', 16:'增持金额'},
    4:{17:'被冻结股东', 18:'冻结开始日期', 19:'冻结结束日期', 20:'冻结金额'},
    5:{21:'质押方', 22:'接收方', 23:'质押开始日期', 24:'质押结束日期', 25:'质押金额'},
    6:{26:'公司名称', 27:'高层人员', 28:'高层职务', 29:'死亡/失联时间', 30:'死亡年龄'},
    7:{31:'公司名称', 32:'公告时间', 33:'损失金额', 34:'其他损失'},
    8:{35:'公司名称', 36:'公告时间', 37:'赔付对象', 38:'赔付金额'}
}

id_2_label = {
    1:'破产清算-公司名称',
    2:'破产清算-公告时间',
    3:'破产清算-公司行业',
    4:'破产清算-裁定时间',
    5:'破产清算-受理法院',
    6:'重大安全事故-公司名称',
    7:'重大安全事故-公告时间',
    8:'重大安全事故-伤亡人数',
    9:'重大安全事故-损失金额',
    10:'重大安全事故-其他影响',
    11:'股东减持-减持的股东',
    12:'股东减持-减持开始日期',
    13:'股东减持-减持金额',
    14:'股东增持-增持的股东',
    15:'股东增持-增持开始日期',
    16:'股东增持-增持金额',
    17:'股权冻结-被冻结股东',
    18:'股权冻结-冻结开始日期',
    19:'股权冻结-冻结结束日期',
    20:'股权冻结-冻结金额',
    21:'股权质押-质押方',
    22:'股权质押-接收方',
    23:'股权质押-质押开始日期',
    24:'股权质押-质押结束日期',
    25:'股权质押-质押金额',
    26:'高层死亡-公司名称',
    27:'高层死亡-高层人员',
    28:'高层死亡-高层职务',
    29:'高层死亡-死亡/失联时间',
    30:'高层死亡-死亡年龄',
    31:'重大资产损失-公司名称',
    32:'重大资产损失-公告时间',
    33:'重大资产损失-损失金额',
    34:'重大资产损失-其他损失',
    35:'重大对外赔付-公司名称',
    36:'重大对外赔付-公告时间',
    37:'重大对外赔付-赔付对象',
    38:'重大对外赔付-赔付金额'
}

class CCKSDataset(Dataset):
    def __init__(self, data_path, config, split='train'):
        self.config = config
        self.data_path=data_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)
        self.raw_dataset = self.__readjson()
        print('原始 {} 数据集长度：'.format(split.upper()),len(self.raw_dataset))
        self.processed_dataset = []
        self.__process()
        print('处理后 {} 数据集长度：'.format(split.upper()),len(self.processed_dataset))


        #for k, v in self.data_list.items():
        #    print(k, ':', v)

    def __readjson(self):
        dataset=[]
        with open(self.data_path,'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset

    def __process(self):
        self.processed_dataset = []
        for idx_in_raw_dataset in track(range(len(self.raw_dataset)), description='Type Processing ...'):
            raw_data = self.raw_dataset[idx_in_raw_dataset]
            # * 预处理

            sentences_num= len(raw_data['combine_id_list'])
            doc_id = raw_data['doc_id']
            event_id = EVENT_TYPE.index(raw_data['events'][0]['event_type'])

            #  得到 content
            sentences_list = []
            for i in raw_data['combine_length_list']:
                if i > 450:
                    print("ERROR")

            for i in range(sentences_num):
                sub_sen_list = []
                for idx in raw_data['combine_id_list'][i]:
                    if type(raw_data['content_list'][idx]) == str:
                        append_content = raw_data['content_list'][idx]
                        sub_sen_list.append(append_content)
                    else:
                        append_content = ''.join([''.join(row) for row in raw_data['content_list'][idx]])
                        sub_sen_list.append(append_content)
                sentences_list.append(''.join(sub_sen_list))
            
            # !判断sentences——list
            for i in range(sentences_num):
                if len(sentences_list[i]) != raw_data['combine_length_list'][i]:
                    print("sentences_list 生成有误！",doc_id, len(sentences_list[i]))

            argument_globalpointer_position = np.zeros([sentences_num, 512, 512]) #torch.zeros(self.config.categories_size,512,512).tolist()
            for event in raw_data['events']:
                for arg_type in event:
                    if arg_type != 'event_type' and arg_type != 'event_id' and event[arg_type] in raw_data['entity_span']:
                        arg_entity = event[arg_type]
                        for position_inform in raw_data['entity_span'][arg_entity]:
                            if position_inform != []:
                                # get prefix
                                for sent_id in range(sentences_num):
                                    if position_inform[0] in raw_data['combine_id_list'][sent_id]:
                                        combined_id = sent_id
                                        break
                                prefix_position = 0
                                for how_long in raw_data['list_length'][raw_data['combine_id_list'][combined_id][0]:position_inform[0]]:
                                    if type(how_long) == list:
                                        prefix_position += sum([sum(long) for long in how_long])
                                    else:
                                        prefix_position += how_long
                                # label
                                argument_globalpointer_position[combined_id][1+prefix_position+position_inform[1]][1+prefix_position+position_inform[2]-1] = ARGUMENT_TYPE_DICT[event_id][arg_type]  #!position_inform[2]要减一！！！
                                #print('argument:',sentences_list[combined_id][prefix_position+position_inform[1]:prefix_position+position_inform[2]])
                                # !!检验
                                span = sentences_list[combined_id][prefix_position+position_inform[1]:prefix_position+position_inform[2]]
                                if span != arg_entity:
                                    print("span匹配有误！！！ ",doc_id,arg_entity,span)

            # *逐条存入数据集
            for i in range(sentences_num):
                data = dict() 
                data['doc_id'] = torch.tensor(int(doc_id))
                data['event_id'] = event_id
                data['events'] = raw_data['events']
                data['content'] = sentences_list[i]
                data['idx_in_content'] = torch.tensor(i)
                data['idx_in_raw_dataset'] = idx_in_raw_dataset

                PTM_input = self.tokenizer.encode_plus(list(data['content']),max_length=512,padding="max_length",is_split_into_words=True)
                data['input_ids'] = torch.tensor(PTM_input['input_ids'])
                data['token_type_ids'] = torch.tensor(PTM_input['token_type_ids'])
                data['attention_mask'] = torch.tensor(PTM_input['attention_mask'])

                data['argument_globalpointer_label'] = torch.from_numpy(argument_globalpointer_position[i]).long()

                self.processed_dataset.append(data)

    def get_table_pos(self,method):
        '''
        method: 
            'default'  不加表格位置编码
            '0-1div'  行和列的pos在0-1间均匀分布
 
        '''
        for idx in range(len(self.processed_dataset)):
            processed_data = self.processed_dataset[idx]
            raw_data = self.raw_dataset[processed_data['idx_in_raw_dataset']]
            idx_in_content = processed_data['idx_in_content']
            table_pos_embeds = [[0.0] * 768 ] * 512

            if self.config.custom_RoPE_pos or self.config.table_self_att or method != 'default':
                # 初始化
                prefix_position = 0
                if self.config.custom_RoPE_pos:                         # // RoPE 
                    RoPE_pos = [[0] * int(self.config.inner_dim/2) ] * 512      # // RoPE 
                    prefix_RoPE_id = 0                                          # // RoPE 
                if self.config.table_self_att:
                    table_self_att = [[False] * 512] * 512

                #逐句处理
                for seg_id in raw_data['combine_id_list'][idx_in_content]:
                    seg = raw_data['content_list'][seg_id]
                    #如果是段落
                    if type(seg) == str:
                        text_length = raw_data['list_length'][seg_id]
                        if self.config.custom_RoPE_pos == True:                 # // RoPE 
                            r_start = 1 + prefix_position                       # // RoPE 
                            r_end = r_start + text_length                       # // RoPE 
                            #====
                            # RoPE_pos[r_start:r_end] = torch.repeat_interleave(torch.arange(prefix_RoPE_id,prefix_RoPE_id+text_length).unsqueeze(1),int(self.config.inner_dim/2),dim=1).tolist()
                            # prefix_RoPE_id += text_length                       # // RoPE 
                            #====
                            # RoPE_pos[r_start:r_end] = [[prefix_RoPE_id,j] * int(self.config.inner_dim/4) for j in range(text_length)] 
                            # prefix_RoPE_id += 1                                 # // RoPE 
                            #====
                            RoPE_pos[r_start:r_end] = [[k,0,0] * int(self.config.inner_dim/6) for k in range(r_start,r_end)] 
                            #====

                        prefix_position += text_length

                    #如果是表格
                    else:
                        assert type(seg) == list
                        table_length = sum([sum(row) for row in raw_data['list_length'][seg_id]])
                        rows = len(seg)
                        cols = len(seg[0])
                        rows_0 = [[0]*2]*cols
                        cols_0 = [[0]*2]*rows

                        #====
                        for i in range(rows):
                            for j in range(cols):
                                pos_in_table = sum([sum(row) for row in raw_data['list_length'][seg_id][0:i]]) \
                                                + sum(raw_data['list_length'][seg_id][i][0:j])
                                cell_length = raw_data['list_length'][seg_id][i][j]
                                p_start = 1 + prefix_position + pos_in_table
                                p_end = p_start + cell_length

                                if self.config.custom_RoPE_pos:        # // RoPE 
                                    # RoPE_pos[p_start:p_end] = [[prefix_RoPE_id+i,prefix_RoPE_id+j] * int(self.config.inner_dim/4) ] * cell_length
                                    RoPE_pos[p_start:p_end] = [[k,i,j] * int(self.config.inner_dim/6) for k in range(p_start,p_end)]

                                if self.config.table_self_att:
                                    if i==0:  #对上表头的注意力
                                        rows_0[j]=[p_start,p_end]
                                    table_self_att[rows_0[j][0]:rows_0[j][1]][p_start:p_end] = [[True] * cell_length] * (rows_0[j][1]-rows_0[j][0])
                                    # if j==0: #对左表头的注意力
                                    #     cols_0[i]=[p_start,p_end]
                                    # table_self_att[cols_0[i][0]:cols_0[i][1]][p_start:p_end] = [[True] * cell_length] * (cols_0[i][1]-cols_0[i][0])


                                # * 不同的编码策略
                                if method == '0-1div':
                                    table_pos_embeds[p_start:p_end] = [[(2.0*i+1)/(2*rows), (2.0*j+1)/(2*cols)] * int(768/2) ] * cell_length

                        if self.config.custom_RoPE_pos:                # // RoPE 
                            prefix_RoPE_id += max(rows,cols)           # // RoPE 

                        prefix_position += table_length

            if self.config.custom_RoPE_pos:                            # // RoPE 
                self.processed_dataset[idx]['RoPE_pos_ids'] = torch.tensor(RoPE_pos)  #int,模型中转为float
            if self.config.table_self_att:
                self.processed_dataset[idx]['table_self_att'] = torch.tensor(table_self_att)
            self.processed_dataset[idx]['table_pos_embeds'] = torch.tensor(table_pos_embeds)  #float

    def __getitem__(self, index):
        from_data = self.processed_dataset[index]
        data = dict()
        data['doc_id'] = from_data['doc_id']
        data['idx_in_content'] = from_data['idx_in_content']
        
        data['input_ids'] = from_data['input_ids']
        data['token_type_ids'] = from_data['token_type_ids']
        data['attention_mask'] = from_data['attention_mask']
        if self.config.custom_RoPE_pos:
            data['RoPE_pos_ids'] = from_data['RoPE_pos_ids']
        if self.config.table_self_att:
            data['table_self_att'] = from_data['table_self_att']
        data['table_pos_embeds'] = from_data['table_pos_embeds']

        data['argument_globalpointer_label']  = from_data['argument_globalpointer_label']     
        return data

    def __len__(self):
        return len(self.processed_dataset)


class CCKSDataOperator(object):
    def __init__(self, config):
        self.config = config
        self.train_path = self.config.data_dir+self.config.train_data
        self.dev_path = self.config.data_dir+self.config.dev_data
        self.test_path = self.config.data_dir+self.config.test_data

#    def __collate_fn(self, batch):
#       return data, label

    def get_dataset(self, split):
        if split == 'train':
            dataset = CCKSDataset(self.train_path,self.config,split='train')
        elif split == 'dev':
            dataset = CCKSDataset(self.dev_path,self.config,split='dev')
        elif split == 'test':
            dataset = CCKSDataset(self.test_path,self.config,split='test')
        return dataset

    def get_dataloader(self, dataset, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            #collate_fn=self.__collate_fn
        )
        return loader



if __name__ == '__main__': #测试用
    from WillMindS.config import Config
    config=Config()
    loader=CCKSDataOperator(config)
    dataset=loader.get_dataset('train',config.checkpoint)
    # dataloader=loader.get_dataloader(dataset)
    # for i, data in enumerate(dataloader):
    #     print(i)
    #     print(data)
    #     break

