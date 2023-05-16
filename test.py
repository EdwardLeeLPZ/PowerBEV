# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import socket
import time

import pytorch_lightning as pl
import torch
from powerbev.config import get_cfg, get_parser
from powerbev.data import prepare_powerbev_dataloaders
from powerbev.trainer import TrainingModule
from pytorch_lightning.plugins import DDPPlugin


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    _, valloader = prepare_powerbev_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH , map_location='cpu'
        )['state_dict']

        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    ) 
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
    )
    trainer.test(model, valloader)


if __name__ == "__main__":
    main()