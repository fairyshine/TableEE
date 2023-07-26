#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import glob
import sys
sys.dont_write_bytecode = True

from rich.progress import track
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from WillMindS.config import Config
from WillMindS.log import Log_Init

from Getdata import CCKSDataset,CCKSDataOperator
from Modelization import PTM_globalpointer,MetricsCalculator

from fill_argument_table.go_with_rule import fill_event_table  #事件表填充，基于规则

class framework_run(object):
    def __init__(self, config, logger):
        self.model = PTM_globalpointer(config).to(config.device)
        logger.info('--------------------------------------')
        logger.info('模型参数信息如下：')
        for _,param in enumerate(self.model.named_parameters()):
            logger.info("{} requires_grad {}".format(param[0], 'TRUE' if param[1].requires_grad == True else 'FALSE'))

        self.data_operator=CCKSDataOperator(config)

        # pickle_path = config.data_dir + config.pickle_path

        logger.info('------------------------------------')
        logger.info('start to process data  ...')
        if config.TRAIN_MODE:
            logger.info('【split - TRAIN】')
            self.train_dataset = self.data_operator.get_dataset('train')
            logger.info('【split - DEV】')
            self.dev_dataset = self.data_operator.get_dataset('dev')
        logger.info('【split - TEST】')
        self.test_dataset = self.data_operator.get_dataset('test')
            
        #     if not os.path.exists(pickle_path):
        #         os.makedirs(pickle_path)
        #     if config.TRAIN_MODE:
        #         torch.save(train_dataset,pickle_path+'TRAIN.pt')
        #         torch.save(dev_dataset,pickle_path+'DEV.pt')
        #         # with open(pickle_path+'TRAIN.pickle', 'wb') as f:
        #         #     pickle.dump(train_dataset, f)
        #         # with open(pickle_path+'DEV.pickle', 'wb') as f:
        #         #     pickle.dump(dev_dataset, f)
        #     torch.save(test_dataset,pickle_path+'TEST.pt')
        #     # with open(pickle_path+'TEST.pickle', 'wb') as f:
        #     #     pickle.dump(test_dataset, f)
        #     logger.info("数据集pickle文件保存完成！目录："+pickle_path)
            
        # logger.info('------------------------------------')
        # logger.info('start to load data  ...')
        # if config.TRAIN_MODE:
        #     logger.info('【split - TRAIN】')
        #     self.train_dataset = torch.load(pickle_path+'TRAIN.pt')
        #     self.train_dataset.config = config
        #     # with open(pickle_path+'TRAIN.pickle', 'rb') as f:
        #     #     self.train_dataset = pickle.load(f)
        #     #     self.train_dataset.config = config
        #     logger.info('【split - DEV】')
        #     self.dev_dataset = torch.load(pickle_path+'DEV.pt')
        #     self.dev_dataset.config = config
        #     # with open(pickle_path+'DEV.pickle', 'rb') as f:
        #     #     self.dev_dataset = pickle.load(f)
        #     #     self.dev_dataset.config = config
        # logger.info('【split - TEST】')
        # self.test_dataset = torch.load(pickle_path+'TEST.pt')
        # self.test_dataset.config = config
        # # with open(pickle_path+'TEST.pickle', 'rb') as f:
        # #     self.test_dataset = pickle.load(f)
        # #     self.test_dataset.config = config

        self.boardwriter = SummaryWriter('log/tensorboard/'+config.log_file+'/')
        
    def train(self):
        logger.info('--------------------------------------')
        logger.info('start to train the model  ...')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.lr_scheduler)
        logger.info('caculate table position embeddings  ...')
        self.train_dataset.get_table_pos(config.table_pos_type)
        self.dev_dataset.get_table_pos(config.table_pos_type)
        train_loader = self.data_operator.get_dataloader(self.train_dataset)
        dev_loader=self.data_operator.get_dataloader(self.dev_dataset)
        best_dev_score = 0.2
        metrics = MetricsCalculator()
        if not os.path.exists(config.output_dir+config.log_file+'/'):
            os.makedirs(config.output_dir+config.log_file+'/')

        for epoch in range(1, config.train_epoch+1):
            logger.info('----------------EPOCH {} ----------------'.format(epoch))

            # * TRAIN 
            train_f1 = 0
            all_steps = len(train_loader)
            self.model.train()
            train_loss = 0
            for step, data in track(enumerate(train_loader),description='Training epoch {} ...'.format(epoch)):

                for key,_ in data.items():
                    if type(data[key]) == torch.Tensor and key !='RoPE_pos_ids':
                        data[key] = data[key].to(config.device)

                out = self.model(data)
                if config.gradient_accumulation_steps > 1:
                    out['loss'] = out['loss'] / config.gradient_accumulation_steps
                out['loss'].backward()

                if step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    self.optimizer.step()
                    self.model.zero_grad()

                batch_f1 = metrics.get_sample_f1(out['logits'], out['label'])

                self.boardwriter.add_scalars('train',{'batch_F1':batch_f1,\
                                                        'batch_loss':out['loss']}, step)
                self.boardwriter.close()

                train_f1 += batch_f1
                train_loss += out['loss'].item()
                logger.info('step:{}/{}   '.format(step+1,all_steps)+'loss:'+str(out['loss'].item())+' F1:'+str(batch_f1))

            logger.info('TRAIN F1:'+str(train_f1/len(train_loader)))
            logger.info('TRAIN loss:'+str(train_loss))

            # * EVAL  =========================================================================
            dev_f1 = 0
            dev_p =0
            dev_r = 0
            self.model.eval()
            with torch.no_grad():
                for step, data in track(enumerate(dev_loader),description='Evaling ...'):

                    for key,_ in data.items():
                        if type(data[key]) == torch.Tensor and key !='RoPE_pos_ids':
                            data[key] = data[key].to(config.device)

                    out = self.model(data)

                    batch_f1,batch_p,batch_r = metrics.get_evaluate_fpr(out['logits'], out['label'])

                    self.boardwriter.add_scalars('dev', {'batch_F1':batch_f1,\
                                                          'batch_P':batch_p,\
                                                          'batch_R':batch_r,\
                                                        'batch_loss':out['loss']}, step)
                    self.boardwriter.close()

                    dev_f1 += batch_f1
                    dev_p += batch_p
                    dev_r += batch_r

                logger.info('DEV F1:'+str(dev_f1/len(dev_loader)))
                logger.info('DEV P :'+str(dev_p/len(dev_loader)))
                logger.info('DEV R :'+str(dev_r/len(dev_loader)))
                logger.info('DEV loss:'+str(out['loss'].item()))

            torch.save(self.model.state_dict(), config.output_dir+config.log_file+'/epoch_{}_devF1_{}.pt'.format(epoch,int(dev_f1/len(dev_loader)*1000000)))


    def test(self,checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
        metrics = MetricsCalculator()

        fault_list = [] #统计事件类型判断错误的doc_id

        logger.info('caculate table position embeddings  ...')
        self.test_dataset.get_table_pos(config.table_pos_type)
        test_loader=self.data_operator.get_dataloader(self.test_dataset)

        test_f1 = 0
        test_p = 0
        test_r = 0

        count_dict = dict()
        with torch.no_grad():
            for _ , data in track(enumerate(test_loader),description='Testing (AE) ...'):

                for key,_ in data.items():
                    if type(data[key]) == torch.Tensor and key !='RoPE_pos_ids':
                        data[key] = data[key].to(config.device)

                out = self.model(data)

                batch_f1,batch_p,batch_r = metrics.get_evaluate_fpr(out['logits'], out['label'])
                test_f1 += batch_f1
                test_p += batch_p
                test_r += batch_r

                for i in range(len(out['predicts'])):
                    doc_id = str(int(data['doc_id'][i]))
                    idx_in_content = int(data['idx_in_content'][i])
                    if doc_id not in count_dict:
                        count_dict[doc_id] = dict()
                    count_dict[doc_id][idx_in_content] = out['predicts'][i]
            
            for key in count_dict:
                count_dict[key] = {idx:count_dict[key][idx] for idx in sorted(count_dict[key])}
                print(count_dict[key])
                # {0: [(42, 56, '高层死亡-公司名称'), (144, 158, '高层死亡-公司名称'), (422, 436, '高层死亡-公司名称'), (174, 177, '高层死亡-高层人员'), (186, 189, '高层死亡-高层人员'), (233, 236, '高层死亡-高层人员'), (280, 283, '高层死亡-高层人员'), (297, 300, '高层死亡-高层人员'), (58, 64, '高层死亡-高层职务'), (168, 174, '高层死亡-高层职务'), (356, 362, '高层死亡-高层职务')]}

            logger.info('TEST F1:'+str(test_f1/len(test_loader)))
            logger.info('TEST P :'+str(test_p/len(test_loader)))
            logger.info('TEST R :'+str(test_r/len(test_loader)))

        # * rule #######################################################
        #pred评价
        pred_dict = dict()
        for data in self.test_dataset.raw_dataset:
            doc_id = data['doc_id']
            if doc_id in count_dict:
                data_input = dict()
                data_input['doc_id'] = doc_id

                evt_type_count = dict()
                for key in count_dict[doc_id]:
                    span_list = count_dict[doc_id][key]
                    for span in span_list:
                        evt_type = span[2].split('-')[0]
                    if evt_type not in evt_type_count:
                        evt_type_count[evt_type] = 1
                    else:
                        evt_type_count[evt_type] += 1

                data_input['event_type_pred'] = max(evt_type_count, key=lambda x: evt_type_count[x])
                
                data_input['event_role_pred'] = []
                data_input['original'] = []

                for i in range(len(data['combine_id_list'])):
                    sub_sen_list = []
                    for idx in data['combine_id_list'][i]:
                        if type(data['content_list'][idx]) == str:
                            append_content = data['content_list'][idx]
                            sub_sen_list.append(append_content)
                        else:
                            append_content = ''.join([''.join(row) for row in data['content_list'][idx]])
                            sub_sen_list.append(append_content)
                    data_input['original'].append(''.join(sub_sen_list))

                data_input['sentences'] = data_input['original']

                for idx_in_content in count_dict[doc_id]:
                    data_input['event_role_pred'].append([idx_in_content,count_dict[doc_id][idx_in_content]])

                pred_dict[doc_id] = fill_event_table(data_input)['events']

                # * for debug
                del data_input['original']
                del data_input['sentences']
                print("规则输入：",data_input)
                print("规则输出：",pred_dict[doc_id])

        # * ############################################################

        COUNT_exact = 0
        COUNT_P = 0
        COUNT_R = 0
        correct_classified_argument = [0,0,0] #* 0-表格，1-混合，2-文本 
        all_classified_argument = [0,0,0]

        for gold_data in self.test_dataset.raw_dataset:
            doc_id = gold_data['doc_id']
            argument_classified_dict = dict()
            for argument in gold_data['entity_span']:
                count = [0,0]
                for position in gold_data['entity_span'][argument]:
                    if type(gold_data['content_list'][position[0]]) == list:
                        count[0] += 1
                    else:
                        assert type(gold_data['content_list'][position[0]]) == str
                        count[1] += 1
                if count[0] > 0:
                    if count[1] > 0:
                        argument_classified_dict[argument] = 1
                    else:
                        argument_classified_dict[argument] = 0
                else:
                    argument_classified_dict[argument] = 2
            for event in gold_data['events']:
                for key in event:
                    if key != 'event_id' and key != 'event_type':
                        all_classified_argument[argument_classified_dict[event[key]]] += 1

            COUNT_R += sum([len(event)-2 for event in gold_data['events']]) # * -2是去掉event_id 和 event_type 
            if doc_id in pred_dict:
                COUNT_P += sum([len(event)-1 for event in pred_dict[doc_id] if event != None]) # * -1是去掉event_type
                if doc_id not in fault_list:
                    print("doc_id:",doc_id)
                    print("gold_event",gold_data['events'])
                    print("pred_event",pred_dict[doc_id])
                    import copy
                    gold_event_list = copy.deepcopy(gold_data['events'])
                    for pred_event in pred_dict[doc_id]:
                        if pred_event != None:
                            if len(gold_event_list) > 0:
                                match_score = [0]*len(gold_event_list)
                                for i in range(len(gold_event_list)):
                                    for key in gold_event_list[i]:
                                        if key != 'event_id' and key != 'event_type':
                                            if key in pred_event:
                                                if pred_event[key] == gold_event_list[i][key]:
                                                    match_score[i] += 1
                                match_index = match_score.index(max(match_score))
                                for key in gold_event_list[match_index]:
                                    if key != 'event_id' and key != 'event_type':
                                        if key in pred_event:
                                            if pred_event[key] == gold_event_list[match_index][key]:
                                                correct_classified_argument[argument_classified_dict[pred_event[key]]]+=1                              
                                COUNT_exact += match_score[match_index]
                                gold_event_list.pop(match_index)
                            else:
                                break

        if COUNT_P * COUNT_R * COUNT_exact > 0:
            P = 1.0*COUNT_exact/COUNT_P
            R = 1.0*COUNT_exact/COUNT_R
            F1 = 2*P*R/(P+R)
            logger.info('TEST SCORE (共{}条测试集数据)'.format(len(pred_dict)))
            logger.info('    - P:'+str(P))
            logger.info('    - R:'+str(R))
            logger.info('    - F1:'+str(F1))
            logger.info(str(correct_classified_argument))
            logger.info(str(all_classified_argument))
            logger.info(str([1.0*correct_classified_argument[i]/all_classified_argument[i] for i in range(3)]))

        return F1 

    def test_one(self,checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
        metrics = MetricsCalculator()

        fault_list = [] #统计事件类型判断错误的doc_id

        logger.info('caculate table position embeddings  ...')
        self.test_dataset.get_table_pos(config.table_pos_type)
        test_loader=self.data_operator.get_dataloader(self.test_dataset)

        test_f1 = 0
        test_p = 0
        test_r = 0

        count_dict = dict()
        with torch.no_grad():
            for _ , data in track(enumerate(test_loader),description='Testing (AE) ...'):

                for key,_ in data.items():
                    if type(data[key]) == torch.Tensor and key !='RoPE_pos_ids':
                        data[key] = data[key].to(config.device)

                out = self.model(data)

                batch_f1,batch_p,batch_r = metrics.get_evaluate_fpr(out['logits'], out['label'])
                test_f1 += batch_f1
                test_p += batch_p
                test_r += batch_r

                for i in range(len(out['predicts'])):
                    doc_id = str(int(data['doc_id'][i]))
                    idx_in_content = int(data['idx_in_content'][i])
                    if doc_id not in count_dict:
                        count_dict[doc_id] = dict()
                    count_dict[doc_id][idx_in_content] = out['predicts'][i]
            
            for key in count_dict:
                count_dict[key] = {idx:count_dict[key][idx] for idx in sorted(count_dict[key])}
                print(count_dict[key])
                # {0: [(42, 56, '高层死亡-公司名称'), (144, 158, '高层死亡-公司名称'), (422, 436, '高层死亡-公司名称'), (174, 177, '高层死亡-高层人员'), (186, 189, '高层死亡-高层人员'), (233, 236, '高层死亡-高层人员'), (280, 283, '高层死亡-高层人员'), (297, 300, '高层死亡-高层人员'), (58, 64, '高层死亡-高层职务'), (168, 174, '高层死亡-高层职务'), (356, 362, '高层死亡-高层职务')]}

            logger.info('TEST F1:'+str(test_f1/len(test_loader)))
            logger.info('TEST P :'+str(test_p/len(test_loader)))
            logger.info('TEST R :'+str(test_r/len(test_loader)))

        # * rule #######################################################
        #pred评价
        pred_dict = dict()
        for data in self.test_dataset.raw_dataset:
            doc_id = data['doc_id']
            if doc_id in count_dict:
                data_input = dict()
                data_input['doc_id'] = doc_id

                evt_type_count = dict()
                for key in count_dict[doc_id]:
                    span_list = count_dict[doc_id][key]
                    for span in span_list:
                        evt_type = span[2].split('-')[0]
                    if evt_type not in evt_type_count:
                        evt_type_count[evt_type] = 1
                    else:
                        evt_type_count[evt_type] += 1

                data_input['event_type_pred'] = max(evt_type_count, key=lambda x: evt_type_count[x])
                
                data_input['event_role_pred'] = []
                data_input['original'] = []

                for i in range(len(data['combine_id_list'])):
                    sub_sen_list = []
                    for idx in data['combine_id_list'][i]:
                        if type(data['content_list'][idx]) == str:
                            append_content = data['content_list'][idx]
                            sub_sen_list.append(append_content)
                        else:
                            append_content = ''.join([''.join(row) for row in data['content_list'][idx]])
                            sub_sen_list.append(append_content)
                    data_input['original'].append(''.join(sub_sen_list))

                data_input['sentences'] = data_input['original']

                for idx_in_content in count_dict[doc_id]:
                    data_input['event_role_pred'].append([idx_in_content,count_dict[doc_id][idx_in_content]])

                pred_dict[doc_id] = fill_event_table(data_input)['events']

                # * for debug
                del data_input['original']
                del data_input['sentences']
                print("规则输入：",data_input)
                print("规则输出：",pred_dict[doc_id])

        # * ############################################################

        COUNT_exact = 0
        COUNT_P = 0
        COUNT_R = 0

        COUNT_type = 0
        COUNT_P_type = 0
        COUNT_R_type = 0

        COUNT_exact_O = 0
        COUNT_P_O = 0
        COUNT_R_O = 0
        COUNT_exact_M = 0
        COUNT_P_M = 0
        COUNT_R_M = 0

        correct_classified_argument = [0,0,0] # * 0-表格，1-混合，2-文本 
        all_classified_argument = [0,0,0]

        OM_flag = 0

        for gold_data in self.test_dataset.raw_dataset:
            doc_id = gold_data['doc_id']
            event_type = gold_data["events"][0]["event_type"]
            if len(gold_data["events"]) > 1:
                OM_flag = 1
            else:
                OM_flag = 0

            # * 统计事件要素来源：表格/文本 
            argument_classified_dict = dict()
            for argument in gold_data['entity_span']:
                count = [0,0]
                for position in gold_data['entity_span'][argument]:
                    if type(gold_data['content_list'][position[0]]) == list:
                        count[0] += 1
                    else:
                        assert type(gold_data['content_list'][position[0]]) == str
                        count[1] += 1
                if count[0] > 0:
                    if count[1] > 0:
                        argument_classified_dict[argument] = 1
                    else:
                        argument_classified_dict[argument] = 0
                else:
                    argument_classified_dict[argument] = 2
            for event in gold_data['events']:
                for key in event:
                    if key != 'event_id' and key != 'event_type':
                        all_classified_argument[argument_classified_dict[event[key]]] += 1

            COUNT_R += sum([len(event)-2 for event in gold_data['events']]) # * -2是去掉event_id 和 event_type 
            COUNT_R_type += 1
            if OM_flag == 0:
                COUNT_R_O += sum([len(event)-2 for event in gold_data['events']])
            else:
                COUNT_R_M += sum([len(event)-2 for event in gold_data['events']])
            if doc_id in pred_dict:
                COUNT_P += sum([len(event)-1 for event in pred_dict[doc_id] if event != None]) # * -1是去掉event_type
                # COUNT_P_type += len(pred_dict[doc_id])
                if OM_flag == 0:
                    COUNT_P_O += sum([len(event)-1 for event in pred_dict[doc_id] if event != None])
                else:
                    COUNT_P_M += sum([len(event)-1 for event in pred_dict[doc_id] if event != None])

                # * type
                type_set = list()
                for pred_event in pred_dict[doc_id]:
                    if pred_event != None:
                        if pred_event['event_type'] not in type_set:
                            type_set.append(pred_event['event_type'])
                COUNT_P_type += len(type_set)
                if event_type in type_set:
                    COUNT_type += 1

                if doc_id not in fault_list:
                    print("doc_id:",doc_id)
                    print("gold_event",gold_data['events'])
                    print("pred_event",pred_dict[doc_id])
                    import copy
                    gold_event_list = copy.deepcopy(gold_data['events'])
                    for pred_event in pred_dict[doc_id]:
                        if pred_event != None:
                            if len(gold_event_list) > 0:
                                match_score = [0]*len(gold_event_list)
                                for i in range(len(gold_event_list)):
                                    for key in gold_event_list[i]:
                                        if key != 'event_id' and key != 'event_type':
                                            if key in pred_event:
                                                if pred_event[key] == gold_event_list[i][key]:
                                                    match_score[i] += 1
                                match_index = match_score.index(max(match_score))
                                for key in gold_event_list[match_index]:
                                    if key != 'event_id' and key != 'event_type':
                                        if key in pred_event:
                                            if pred_event[key] == gold_event_list[match_index][key]:
                                                correct_classified_argument[argument_classified_dict[pred_event[key]]]+=1                              
                                COUNT_exact += match_score[match_index]
                                if OM_flag == 0:
                                    COUNT_exact_O += match_score[match_index]
                                else:
                                    COUNT_exact_M += match_score[match_index]
                                gold_event_list.pop(match_index)
                            else:
                                break

        if COUNT_P * COUNT_R * COUNT_exact > 0:
            P = 1.0*COUNT_exact/COUNT_P
            R = 1.0*COUNT_exact/COUNT_R
            F1 = 2*P*R/(P+R)
            logger.info('TEST SCORE (共{}条测试集数据)'.format(len(pred_dict)))
            logger.info('    - P:'+str(P))
            logger.info('    - R:'+str(R))
            logger.info('    - F1:'+str(F1))
            logger.info(str(correct_classified_argument))
            logger.info(str(all_classified_argument))
            logger.info(str([1.0*correct_classified_argument[i]/all_classified_argument[i] for i in range(3)]))
            logger.info("======其他结果======")
            logger.info("======类型======")
            P_type = 1.0*COUNT_type/COUNT_P_type
            R_type = 1.0*COUNT_type/COUNT_R_type
            F1_type = 2*P_type*R_type/(P_type+R_type)
            logger.info('    - P:'+str(P_type))
            logger.info('    - R:'+str(R_type))
            logger.info('    - F1:'+str(F1_type))
            logger.info("======事件O2O======")
            P_o = 1.0*COUNT_exact_O/COUNT_P_O
            R_o = 1.0*COUNT_exact_O/COUNT_R_O
            F1_o = 2*P_o*R_o/(P_o+R_o)
            logger.info('    - P:'+str(P_o))
            logger.info('    - R:'+str(R_o))
            logger.info('    - F1:'+str(F1_o))
            logger.info("======事件O2M======")
            P_m = 1.0*COUNT_exact_M/COUNT_P_M
            R_m = 1.0*COUNT_exact_M/COUNT_R_M
            F1_m = 2*P_m*R_m/(P_m+R_m)
            logger.info('    - P:'+str(P_m))
            logger.info('    - R:'+str(R_m))
            logger.info('    - F1:'+str(F1_m))
        return F1

if __name__ == '__main__':
    config = Config()
    logger = Log_Init(config)
    config.log_print_config(logger)

    main_run = framework_run(config, logger)

    # * 训练 
    if config.TRAIN_MODE == True:
        main_run.train()

    # * 测试 
    best_checkpoint = ''
    best_F1 = 0
    if config.TEST_MODE == True:
        # # * 测所有checkpoint  
        # checkpoint_list = glob.glob(config.output_model_dir+'*.pt')
        # checkpoint_list = sorted(checkpoint_list, key=lambda info: (info[0], int(info.split('_')[1])))
        # for checkpoint in checkpoint_list:
        #     logger.info(checkpoint)
        #     try:
        #         F1 = main_run.test(checkpoint)
        #         if F1 > best_F1:
        #             best_F1 = F1
        #             best_checkpoint = checkpoint
        #     except:
        #         pass
        # logger.info("测试完毕！")
        # logger.info("最佳结果：F1：{}   来自：{} ".format(best_F1,best_checkpoint))

        # * 测单个checkpoint  
        checkpoint = config.output_model_dir + config.model_for_test
        logger.info("TEST ONE CHECKPOINT 模式：")
        logger.info(checkpoint)
        _ = main_run.test_one(checkpoint)


