import os
from typing import List, Dict, Type
import math

import torch
from torch.optim import Optimizer
import transformers

WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        dataloader,
        eval_dataloader,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/vision_text_pretrain',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()

        skip_scheduler = False
        for epoch in range(epochs):
            data_iterator = iter(dataloader)
            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)

                if use_amp:
                    with autocast():
                        loss = model(data)
                    loss_value = loss['loss']
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss = model(data)
                    loss_value = loss['loss'] / self.accumulation_steps
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value))
                

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
                    


                if train_iter == (steps_per_epoch-1) and 't5' in output_path:
                    eval_data_iterator = iter(eval_dataloader)
                    num_iter = len(eval_dataloader)
                    for eval_iter in range(num_iter):           
                        eval_data = next(eval_data_iterator)
                        images = eval_data['images'].cuda().half()
                        text = []
                        question = []
                        answer = []
                        tq = []
                        bs = len(eval_data['reports'])
                        for b in range(bs):
                            doc = eval_data['reports'][b] 
                            if 'The diagnosis is' in doc:
                                text.append(doc.split('The diagnosis is ')[0])
                                question.append('What will this subject be diagnosed with?') # hard coded
                                label = doc.split('The diagnosis is ')[1].split('.')[0]
                                label = label.replace('AD','Dementia')
                                label = label.replace('Demented','Dementia')
                                label = label.replace('NC','Not demented')
                                label = label.replace('CN','Not demented')
                                label = label.replace('Nondemented','Not demented')
                                label = label.replace('control','Not demented')
                                label = label.replace('MCI','mild cognitive impairment (MCI)')
                                answer.append(label)
                                tq.append(doc.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')

                            else:
                                text.append(doc)
                                question.append('What will this subject be diagnosed with?') # hard coded
                                answer.append('')
                                tq.append(doc.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')
                        model.eval()
                        res = model.generate({"images": images, 'prompt': tq}) # "images": images, 
                        
                        for i in range(bs):
                            print('eval_iter[{}/{}][{}/{}] report: '.format(eval_iter,num_iter,i,bs), eval_data['reports'][i])
                            print('eval_iter[{}/{}][{}/{}] prompt: '.format(eval_iter,num_iter,i,bs), tq[i])
                            print('eval_iter[{}/{}][{}/{}] gt_answer: '.format(eval_iter,num_iter,i,bs), answer[i])
                            print('eval_iter[{}/{}][{}/{}] answer: '.format(eval_iter,num_iter,i,bs), res[i])
                            print('-----------------------------------------------')

                if train_iter == (1) and 'biomedlm' in output_path:
                    eval_data_iterator = iter(eval_dataloader)
                    num_iter = len(eval_dataloader)
                    for eval_iter in range(num_iter):           
                        eval_data = next(eval_data_iterator)
                        images = eval_data['images'].cuda().half()
                        text = []
                        question = []
                        answer = []
                        tq = []
                        bs = len(eval_data['reports'])
                        for b in range(bs):
                            doc = eval_data['reports'][b] 
                            if 'The diagnosis is' in doc:
                                text.append(doc.split('The diagnosis is ')[0])
                                question.append('What will this subject be diagnosed with?') # hard coded
                                label = doc.split('The diagnosis is ')[1].split('.')[0]
                                label = label.replace('AD','Dementia')
                                label = label.replace('Demented','Dementia')
                                label = label.replace('NC','Not demented')
                                label = label.replace('CN','Not demented')
                                label = label.replace('Nondemented','Not demented')
                                label = label.replace('control','Not demented')
                                label = label.replace('MCI','mild cognitive impairment (MCI)')
                                answer.append(label)
                                tq.append(doc.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')

                            else:
                                text.append(doc)
                                question.append('What will this subject be diagnosed with?') # hard coded
                                answer.append('')
                                tq.append(doc.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')
                        model.eval()
                        res = model.generate({"images": images, 'prompt': tq}) # "images": images, 
                        
                        for i in range(bs):
                            print('eval_iter[{}/{}][{}/{}] report: '.format(eval_iter,num_iter,i,bs), eval_data['reports'][i])
                            print('eval_iter[{}/{}][{}/{}] prompt: '.format(eval_iter,num_iter,i,bs), tq[i])
                            print('eval_iter[{}/{}][{}/{}] gt_answer: '.format(eval_iter,num_iter,i,bs), answer[i])
                            print('eval_iter[{}/{}][{}/{}] answer: '.format(eval_iter,num_iter,i,bs), res[i])
                            print('-----------------------------------------------')

            self._save_ckpt(model,epoch,output_path)


    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, epoch, save_dir):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, 'epoch{}.pth'.format(epoch)))
