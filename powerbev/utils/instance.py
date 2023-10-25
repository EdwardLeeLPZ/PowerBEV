# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from powerbev.utils.geometry import flow_warp
from scipy.optimize import linear_sum_assignment


# set ignore index to 0 for vis
def convert_instance_mask_to_center_and_offset_label(instance_img, num_instances, ignore_index=255, sigma=3):
    seq_len, h, w = instance_img.shape
    center_label = torch.zeros(seq_len, 1, h, w)
    offset_label = ignore_index * torch.ones(seq_len, 2, h, w)
    # x is vertical displacement, y is horizontal displacement
    x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))

    # Ignore id 0 which is the background
    for instance_id in range(1, num_instances+1):
        for t in range(seq_len):
            instance_mask = (instance_img[t] == instance_id)

            xc = x[instance_mask].mean().round().long()
            yc = y[instance_mask].mean().round().long()

            off_x = xc - x
            off_y = yc - y
            g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
            center_label[t, 0] = torch.maximum(center_label[t, 0], g)
            offset_label[t, 0, instance_mask] = off_x[instance_mask]
            offset_label[t, 1, instance_mask] = off_y[instance_mask]

    return center_label, offset_label


def find_instance_centers(center_prediction: torch.Tensor, conf_threshold: float = 0.1, nms_kernel_size: float = 3):
    assert len(center_prediction.shape) == 3
    center_prediction = F.threshold(center_prediction, threshold=conf_threshold, value=-1)

    nms_padding = (nms_kernel_size - 1) // 2
    maxpooled_center_prediction = F.max_pool2d(
        center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
    )

    # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
    center_prediction[center_prediction != maxpooled_center_prediction] = -1
    return torch.nonzero(center_prediction > 0)[:, 1:]


def group_pixels(centers: torch.Tensor, offset_predictions: torch.Tensor) -> torch.Tensor:
    width, height = offset_predictions.shape[-2:]
    x_grid = (
        torch.arange(width, dtype=offset_predictions.dtype, device=offset_predictions.device)
        .view(1, width, 1)
        .repeat(1, 1, height)
    )
    y_grid = (
        torch.arange(height, dtype=offset_predictions.dtype, device=offset_predictions.device)
        .view(1, 1, height)
        .repeat(1, width, 1)
    )
    pixel_grid = torch.cat((x_grid, y_grid), dim=0)
    offset = torch.stack([offset_predictions[1], offset_predictions[0]], dim=0)
    center_locations = (pixel_grid + offset).view(2, width * height, 1).permute(2, 1, 0)
    centers = centers.view(-1, 1, 2)

    distances = torch.norm(centers - center_locations, dim=-1)

    instance_id = torch.argmin(distances, dim=0).reshape(1, width, height) + 1
    return instance_id


def get_instance_segmentation_and_centers(
    center_predictions: torch.Tensor,
    offset_predictions: torch.Tensor,
    foreground_mask: torch.Tensor,
    conf_threshold: float = 0.1,
    nms_kernel_size: float = 5,
    max_n_instance_centers: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    width, height = offset_predictions.shape[-2:]
    center_predictions = center_predictions.view(1, width, height)
    offset_predictions = offset_predictions.view(2, width, height)
    foreground_mask = foreground_mask.view(1, width, height)

    centers = find_instance_centers(center_predictions, conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size)
    if not len(centers):
        return torch.zeros(center_predictions.shape, dtype=torch.int64, device=center_predictions.device)

    if len(centers) > max_n_instance_centers:
        # print(f'There are a lot of detected instance centers: {centers.shape}')
        centers = centers[:max_n_instance_centers].clone()

    instance_ids = group_pixels(centers, offset_predictions * foreground_mask.float())
    instance_seg = (instance_ids * foreground_mask.float()).long()

    # Make the indices of instance_seg consecutive
    instance_seg = make_instance_seg_consecutive(instance_seg)

    return instance_seg.long()


def update_instance_ids(instance_seg, old_ids, new_ids):
    """
    Parameters
    ----------
        instance_seg: torch.Tensor arbitrary shape
        old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
        new_ids: 1D tensor with the new ids, aligned with old_ids

    Returns
        new_instance_seg: torch.Tensor same shape as instance_seg with new ids
    """
    indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id

    return indices[instance_seg].long()


def make_instance_seg_consecutive(instance_seg):
    # Make the indices of instance_seg consecutive
    unique_ids = torch.unique(instance_seg)
    new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg


def make_instance_id_temporally_consecutive(pred_inst, preds, backward_flow, ignore_index=255.0):
    assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

    # Initialise instance segmentations with prediction corresponding to the present
    consistent_instance_seg = [pred_inst[:, 0:1]]
    backward_flow = backward_flow.clone().detach()
    backward_flow[backward_flow == ignore_index] = 0.0
    _, seq_len, _, h, w = preds.shape

    for t in range(1, seq_len):
        init_warped_instance_seg = flow_warp(consistent_instance_seg[-1].unsqueeze(2).float(), backward_flow[:, t:t+1]).squeeze(2).int()

        warped_instance_seg = init_warped_instance_seg * preds[:, t:t+1, 0]
    
        consistent_instance_seg.append(warped_instance_seg)
    
    consistent_instance_seg = torch.cat(consistent_instance_seg, dim=1)
    return consistent_instance_seg


def predict_instance_segmentation(output, compute_matched_centers=False,  vehicles_id=1, spatial_extent=[50, 50]):
    preds = output['segmentation'].detach()
    preds = torch.argmax(preds, dim=2, keepdims=True)
    foreground_masks = preds.squeeze(2) == vehicles_id

    batch_size, seq_len = preds.shape[:2]
    pred_inst = []
    for b in range(batch_size):
        pred_inst_batch = get_instance_segmentation_and_centers(
            torch.softmax(output['segmentation'], dim=2)[b, 0:1, vehicles_id].detach(),
            output['instance_flow'][b, 1:2].detach(),
            foreground_masks[b, 1:2].detach(),
            nms_kernel_size=round(350/spatial_extent[0]),
        )
        pred_inst.append(pred_inst_batch)
    pred_inst = torch.stack(pred_inst).squeeze(2)

    if output['instance_flow'] is None:
        print('Using zero flow because instance_future_output is None')
        output['instance_flow'] = torch.zeros_like(output['instance_flow'])
    consistent_instance_seg = []
    for b in range(batch_size):
        consistent_instance_seg.append(
            make_instance_id_temporally_consecutive(
                pred_inst[b:b+1],
                preds[b:b+1, 1:],
                output['instance_flow'][b:b+1, 1:].detach(),
                )
        )
    consistent_instance_seg = torch.cat(consistent_instance_seg, dim=0)
    consistent_instance_seg = torch.cat([torch.zeros_like(pred_inst), consistent_instance_seg], dim=1)

    if compute_matched_centers:
        assert batch_size == 1
        # Generate trajectories
        matched_centers = {}
        _, seq_len, h, w = consistent_instance_seg.shape
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=preds.device),
            torch.arange(w, dtype=torch.float, device=preds.device)
        ))

        for instance_id in torch.unique(consistent_instance_seg[0, 1])[1:].cpu().numpy():
            for t in range(seq_len):
                instance_mask = consistent_instance_seg[0, t] == instance_id
                if instance_mask.sum() > 0:
                    matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                        grid[:, instance_mask].mean(dim=-1)]

        for key, value in matched_centers.items():
            matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

        return consistent_instance_seg, matched_centers

    return consistent_instance_seg.long()


def generate_gt_instance_segmentation(output, compute_matched_centers=False,  vehicles_id=1, spatial_extent=[50, 50]):
    preds = output['segmentation']
    foreground_masks = preds.squeeze(2) == vehicles_id
    output['segmentation'] = output['segmentation'].float() + output['centerness'].float()

    batch_size, seq_len = preds.shape[:2]
    pred_inst = []
    for b in range(batch_size):
        pred_inst_batch = get_instance_segmentation_and_centers(
            output['segmentation'][b, 0:1, 0].detach(),
            output['instance_flow'][b, 1:2].detach(),
            foreground_masks[b, 1:2].detach(),
            nms_kernel_size=round(350/spatial_extent[0]),
        )
        pred_inst.append(pred_inst_batch)
    pred_inst = torch.stack(pred_inst).squeeze(2)

    if output['instance_flow'] is None:
        print('Using zero flow because instance_future_output is None')
        output['instance_flow'] = torch.zeros_like(output['instance_flow'])
    consistent_instance_seg = []
    for b in range(batch_size):
        consistent_instance_seg.append(
            make_instance_id_temporally_consecutive(
                pred_inst[b:b+1],
                preds[b:b+1, 1:],
                output['instance_flow'][b:b+1, 1:].detach(),
                )
        )
    consistent_instance_seg = torch.cat(consistent_instance_seg, dim=0)
    consistent_instance_seg = torch.cat([torch.zeros_like(pred_inst), consistent_instance_seg], dim=1)

    if compute_matched_centers:
        assert batch_size == 1
        # Generate trajectories
        matched_centers = {}
        _, seq_len, h, w = consistent_instance_seg.shape
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=preds.device),
            torch.arange(w, dtype=torch.float, device=preds.device)
        ))

        for instance_id in torch.unique(consistent_instance_seg[0, 1])[1:].cpu().numpy():
            for t in range(seq_len):
                instance_mask = consistent_instance_seg[0, t] == instance_id
                if instance_mask.sum() > 0:
                    matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                        grid[:, instance_mask].mean(dim=-1)]

        for key, value in matched_centers.items():
            matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

        return consistent_instance_seg, matched_centers

    return consistent_instance_seg.long()