import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.blip2_models.blip2 import Blip2Base
from medblip.modeling_gpt2 import GPT2LMHeadModel
from medblip.eva_vit import create_eva_vit_g
from transformers import GPT2Tokenizer

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MedBLIPModel_biomedlm(Blip2Base):
    """
    BLIP2 t5
    """
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        patch_size=32,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        lm_model="stanford-crfm/BioMedLM",
        prompt="",
        max_txt_len=100,
        apply_lemmatizer=False,
        embed_dim=256,
    ):
        super().__init__()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, patch_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if '3d' not in name:
                    param.requires_grad = False

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None


        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_model, pad_token='<PAD>')
        self.lm_model = GPT2LMHeadModel.from_pretrained(lm_model, torch_dtype=torch.float16)

        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.qa_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.proj = nn.Linear(self.Qformer.config.hidden_size, self.lm_model.config.n_embd)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len

    def init_vision_encoder(
        cls, 
        model_name, 
        img_size, 
        patch_size,
        drop_path_rate, 
        use_grad_checkpoint, 
        precision
    ):
        visual_encoder = create_eva_vit_g(
                img_size,
                patch_size, 
                drop_path_rate, 
                use_grad_checkpoint, 
                precision
            )
        
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def forward(self, samples):
        image = samples["images"].cuda().half()
        text = []
        question = []
        answer = []
        qa = []
        tq = []

        bs = len(samples['reports'])
        for b in range(bs):
            doc = samples['reports'][b] 
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
                qa.append('Question: What will this subject be diagnosed with? Answer: ' + label)
                tq.append(doc.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')

            else:
                text.append(doc)
                question.append('What will this subject be diagnosed with?') # hard coded
                answer.append('')
                qa.append('Question: What will this subject be diagnosed with? Answer: ')
                tq.append(doc.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')


        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids, 
            attention_mask=text_tokens.attention_mask, 
            return_dict=True,)

        
        qa_tokens = self.tokenizer(
            qa,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",).to(image.device)
        qa_output = self.Qformer.bert(
            qa_tokens.input_ids, 
            attention_mask=qa_tokens.attention_mask, 
            return_dict=True,)
        
        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        qa_feat = F.normalize(self.qa_proj(qa_output.last_hidden_state[:, 0, :]), dim=-1)

        ##################################### ITC ########################################

        sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat.unsqueeze(-1)).squeeze()
        sim_i2t, index_i2t = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.matmul(text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp


        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        sim_q2t = torch.matmul(image_feats.unsqueeze(1), qa_feat.unsqueeze(-1)).squeeze()
        sim_i2t, index_i2t = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.matmul(qa_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp


        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)

        loss_itc += (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2



        img_embeds = self.proj(query_output.last_hidden_state) # bs 32 
        atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(image.device)

        self.tokenizer.padding_side = "right"

        input_tokens = self.tokenizer(
            tq,
            padding="longest",
            truncation=True, 
            max_length=self.max_txt_len, 
            return_tensors="pt",).to(image.device)
        output_tokens = self.tokenizer(
            answer,
            padding="longest",
            truncation=True, 
            max_length=self.max_txt_len, 
            return_tensors="pt",).to(image.device)
        
        input_targets = input_tokens.input_ids.masked_fill(
            input_tokens.input_ids == self.tokenizer.pad_token_id, -100) # bs output_txt_len(<=max_len) 0的位置填-100
        output_targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.tokenizer.pad_token_id, -100)
        
        empty_img_targets = (
            torch.ones(atts_img.size(), dtype=torch.long).to(image.device).fill_(-100) # bs 32
        )

        targets = torch.cat([input_targets, empty_img_targets, output_targets], dim=1) # bs 32+txt_len (这里的32都是-100不计算loss)

        inputs_txt_embeds = self.lm_model.transformer.wte(input_tokens.input_ids) # bs txt_len 2560
        outputs_txt_embeds = self.lm_model.transformer.wte(output_tokens.input_ids) # bs txt_len 2560
        inputs_embeds = torch.cat([inputs_txt_embeds,img_embeds,outputs_txt_embeds], dim=1)

        attention_mask = torch.cat([input_tokens.attention_mask,atts_img,output_tokens.attention_mask], dim=1) # bs 32+input_txt_len

        with self.maybe_autocast():
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,)

        loss_lm = outputs.loss
        loss=loss_itc+loss_lm
        print('loss_itc', loss_itc, 'loss_lm', outputs.loss)
        pred = self.tokenizer.batch_decode(outputs['logits'].argmax(-1), skip_special_tokens=True)

        for i in range(bs):
            print('train_iter[{}/{}] text: '.format(i,bs), text[i])
            print('train_iter[{}/{}] question: '.format(i,bs), question[i])
            print('train_iter[{}/{}] gt_answer: '.format(i,bs), answer[i])
            print('train_iter[{}/{}] answer: '.format(i,bs), pred[i])
            print('-----------------------------------------------')

        return {"loss": loss}

    
    @torch.no_grad()
    def generate(
        self,
        samples,
        device='cuda:0',
    ):
        
        input_tokens = self.tokenizer(
            samples["prompt"], 
            padding="longest",
            truncation=True, 
            return_tensors="pt",
            max_length=self.max_txt_len).to(device)
        
        image = samples["images"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_img = self.proj(query_output.last_hidden_state)
        atts_img = torch.ones(inputs_img.size()[:-1], dtype=torch.long).to(image.device)


        attention_mask = torch.cat([atts_img,input_tokens.attention_mask], dim=1)

        inputs_txt_embeds = self.lm_model.transformer.wte(input_tokens.input_ids) 
        inputs_embeds = torch.cat([inputs_txt_embeds,inputs_img],dim=1)
        attention_mask = torch.cat([input_tokens.attention_mask, atts_img], dim=1)
        filler_input_ids = torch.ones([inputs_embeds.shape[0],1], dtype=torch.long).to(image.device).fill_(self.lm_model.config.bos_token_id).to(image.device)

        with self.maybe_autocast():
            outputs = self.lm_model.generate(
                filler_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask= attention_mask,
                )
        
        output_text = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text