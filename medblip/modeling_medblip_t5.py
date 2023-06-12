import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from torch.nn import functional as F

from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from medblip.eva_vit import create_eva_vit_g

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MedBLIPModel_t5(Blip2Base):
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
        t5_model="google/flan-t5-xl",
        max_txt_len=60,
        embed_dim=256,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
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

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        t5_config.output_attentions = True
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.qa_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
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



        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                tq,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([input_tokens.attention_mask,atts_t5], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

            inputs_embeds = torch.cat([inputs_embeds,inputs_t5], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                labels=targets,
            )

            loss_lm = outputs.loss
            loss=loss_itc+loss_lm
            print('loss_itc', loss_itc, 'loss_lm', outputs.loss)



            pred = self.t5_tokenizer.batch_decode(outputs['logits'].argmax(-1), skip_special_tokens=True)

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
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=60,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        device='cuda:0',
    ):
        # import pdb;pdb.set_trace()
        
        input_tokens = self.t5_tokenizer(
            samples["prompt"], 
            padding="longest", 
            return_tensors="pt").to(device)
        
        if 'images' in samples.keys():
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
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            encoder_atts = torch.cat([input_tokens.attention_mask,atts_t5], dim=1)
            with self.maybe_autocast(dtype=torch.bfloat16):
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_embeds,inputs_t5], dim=1)

        else:
            encoder_atts = input_tokens.attention_mask
            with self.maybe_autocast(dtype=torch.bfloat16):
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                
        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            return_dict_in_generate=True
        )
        output_text = self.t5_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return output_text
    