# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

import os
from argparse import ArgumentParser
from glob import glob

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from powerbev.config import get_cfg, get_parser
from powerbev.data import prepare_powerbev_dataloaders
from powerbev.trainer import TrainingModule
from powerbev.utils.instance import predict_instance_segmentation, generate_gt_instance_segmentation
from powerbev.utils.network import NormalizeInverse
from powerbev.utils.visualisation import (convert_figure_numpy,
                                          generate_instance_colours,
                                          make_contour, plot_instance_map)


def plot_prediction(image, output, cfg):
    if cfg.VISUALIZATION.VIS_GT:
        # Process ground truth
        consistent_instance_seg, matched_centers = generate_gt_instance_segmentation(
            output, compute_matched_centers=True, spatial_extent=(cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        )
    else:
        # Process predictions
        consistent_instance_seg, matched_centers = predict_instance_segmentation(
            output, compute_matched_centers=True, spatial_extent=(cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        )
    first_instance_seg = consistent_instance_seg[0, 1]

    # Plot future trajectories
    unique_ids = torch.unique(first_instance_seg).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(first_instance_seg.cpu().numpy(), instance_map)
    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        for t in range(len(path) - 1):
            color = instance_colours[instance_id].tolist()
            cv2.line(trajectory_img, tuple(path[t]), tuple(path[t + 1]),
                     color, 4)

    # Overlay arrows
    temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
    mask = ~ np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]

    # Plot present RGB frames and predictions
    val_w = 2.99
    cameras = cfg.IMAGE.NAMES
    image_ratio = cfg.IMAGE.FINAL_DIM[0] / cfg.IMAGE.FINAL_DIM[1]
    val_h = val_w * image_ratio
    fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
    width_ratios = (val_w, val_w, val_w, val_w)
    gs = mpl.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )
    for imgi, img in enumerate(image[0, -1]):
        ax = plt.subplot(gs[imgi // 3, imgi % 3])
        showimg = denormalise_img(img.cpu())
        if imgi > 2:
            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

        plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                     xycoords='axes fraction', fontsize=14)
        plt.imshow(showimg)
        plt.axis('off')

    ax = plt.subplot(gs[:, 3])
    plt.imshow(make_contour(vis_image[::-1, ::-1]))
    plt.axis('off')

    plt.draw()
    figure_numpy = convert_figure_numpy(fig)
    plt.close()
    return figure_numpy
    

def visualise():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    _, valloader = prepare_powerbev_dataloaders(cfg)

    trainer = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        weights_path = cfg.PRETRAINED.PATH
        pretrained_model_weights = torch.load(
            weights_path , map_location='cpu'
        )['state_dict']

        trainer.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {weights_path}')

    device = torch.device('cuda:0')
    trainer = trainer.to(device)
    trainer.eval()

    for i, batch in enumerate(valloader):
        if cfg.VISUALIZATION.VIS_GT:
            # Visualize ground truth
            image = batch['image'].to(device)
            time_range = cfg.TIME_RECEPTIVE_FIELD - 2
            output = {
                'segmentation': batch['segmentation'][:, time_range:].to(device),
                'instance_flow': batch['flow'][:, time_range:].to(device),
                'centerness': batch['centerness'][:, time_range:].to(device),
            }

            figure_numpy = plot_prediction(image, output, cfg)
        else:
            # Visualize predictions
            image = batch['image'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            future_egomotions = batch['future_egomotion'].to(device)

            # Forward pass
            with torch.no_grad():
                output = trainer.model(image, intrinsics, extrinsics, future_egomotions)
            
            figure_numpy = plot_prediction(image, output, cfg)
        
        os.makedirs(os.path.join(cfg.VISUALIZATION.OUTPUT_PATH), exist_ok=True)
        output_filename = os.path.join(cfg.VISUALIZATION.OUTPUT_PATH, 'sample_'+str(i)) + '.png'
        Image.fromarray(figure_numpy).save(output_filename)
        print(f'Saved output in {output_filename}')

        if i >= cfg.VISUALIZATION.SAMPLE_NUMBER-1:
            return


if __name__ == '__main__':
    visualise()