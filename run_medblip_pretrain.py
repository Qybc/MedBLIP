import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from medblip.modeling_medblip_t5 import MedBLIPModel_t5
from medblip.modeling_medblip_biomedlm import MedBLIPModel_biomedlm
from medblip.dataset import ImageTextContrastiveDataset,ZeroShotImageDataset
from medblip.dataset import ImageTextContrastiveCollator,ZeroShotImageCollator
from medblip.trainer import Trainer

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_config = {
    'num_epochs': 100,
    'warmup': 0.1,
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'eval_batch_size': 8,
    'eval_steps': 1000,
    'save_steps': 1000,
}

train_datalist = [
    'ADNI-train',
    'NACC-train',
    'OASIS1-aligned_norm-train',
    'OASIS1-aligned_orig-train',
    'OASIS1-norm-train',
    'OASIS1-orig-train',
    'OASIS2-train',
]

val_datalist = [
    # 'ADNI-test',
    'NACC-test',
    # 'OASIS2-test',
    # 'AIBL-test',
    # 'MIRIAD-test',
]

traindata = ImageTextContrastiveDataset(datalist=train_datalist)
train_collate_fn = ImageTextContrastiveCollator()
trainloader = DataLoader(traindata,
    batch_size=7,
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
    )

val_data = ZeroShotImageDataset(datalist=val_datalist)
val_collate_fn = ZeroShotImageCollator()
valloader = DataLoader(val_data,
    batch_size=7,
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    )



t5=False
biomedlm=True


if t5:
    model = MedBLIPModel_t5(
        t5_model="google/flan-t5-xl",
    )
    # model.load_state_dict(torch.load('./checkpoints/vision_text_pretrain/t5/epoch10.pth',map_location='cpu'),strict=False)
    model.cuda()
    model_save_path = f'./checkpoints/vision_text_pretrain/t5'
    trainer = Trainer()
    trainer.train(
        model,
        trainloader,
        valloader,
        warmup_ratio=train_config['warmup'],
        epochs=train_config['num_epochs'],
        optimizer_params={'lr':train_config['lr']},
        output_path=model_save_path,
        weight_decay=train_config['weight_decay'],
        use_amp=True,
        )

if biomedlm:
    model = MedBLIPModel_biomedlm(
        lm_model="stanford-crfm/BioMedLM",
    )
    # model.load_state_dict(torch.load('./checkpoints/vision_text_pretrain/biomedlm/epoch11.pth',map_location='cpu'),strict=False)
    model.cuda()
    model_save_path = f'./checkpoints/vision_text_pretrain/biomedlm'
    trainer = Trainer()
    trainer.train(
        model,
        trainloader,
        valloader,
        warmup_ratio=train_config['warmup'],
        epochs=train_config['num_epochs'],
        optimizer_params={'lr':train_config['lr']},
        output_path=model_save_path,
        weight_decay=train_config['weight_decay'],
        use_amp=True,
        )
    

