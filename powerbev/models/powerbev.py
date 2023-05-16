# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

import time

import torch
import torch.nn as nn
from powerbev.models.decoder import Decoder
from powerbev.models.encoder import Encoder
from powerbev.models.stconv import MultiBranchSTconv
from powerbev.models.temporal_model import TemporalModelIdentity
from powerbev.utils.geometry import (VoxelsSumming,
                                     calculate_birds_eye_view_parameters,
                                     cumulative_warp_features)
from powerbev.utils.network import (pack_sequence_dim, set_bn_momentum,
                                    unpack_sequence_dim)


class PowerBEV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        if self.cfg.TIME_RECEPTIVE_FIELD == 1:
            assert self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity'

        # temporal block
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_future = self.cfg.N_FUTURE_FRAMES

        if self.cfg.MODEL.SUBSAMPLE:
            assert self.cfg.DATASET.NAME == 'lyft'
            self.receptive_field = 3
            self.n_future = 5

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)

        # Temporal model
        temporal_in_channels = self.encoder_out_channels
        temporal_out_channels = temporal_in_channels    
        if self.n_future == 0:
            self.temporal_model = TemporalModelIdentity(temporal_in_channels, self.receptive_field)
            # Decoder
            self.future_pred_in_channels = temporal_out_channels
            self.decoder = Decoder(
                in_channels=self.future_pred_in_channels,
                n_classes=len(self.cfg.SEMANTIC_SEG.WEIGHTS),
                predict_future_flow=self.cfg.INSTANCE_FLOW.ENABLED,
            )
        elif self.n_future > 0:
            if self.cfg.MODEL.STCONV.INPUT_EGOPOSE:
                temporal_in_channels += 6
            # Multi-branch STConv
            self.stconv = MultiBranchSTconv(cfg=self.cfg, in_channels = temporal_in_channels)
        else:
            raise NotImplementedError(f'Number of future frames {self.n_future}.')

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs=None, noise=None):
        output = {}
        start_time = time.time()

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()

        # Lifting features and project to bird's-eye view
        x = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics)

        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        perception_time = time.time()
       
        # Temporal model
        if self.n_future == 0:
            states = self.temporal_model(x)
            bev_output = self.decoder(states[:, -1:])
        elif self.n_future > 0:
            if self.cfg.MODEL.STCONV.INPUT_EGOPOSE:
                b, s, c = future_egomotion.shape
                h, w = x.shape[-2:]
                future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
                # at time 0, no egomotion so feed zero vector
                future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                    future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
                x = torch.cat([x, future_egomotions_spatial], dim=-3)

            bev_output = self.stconv(x)
        else:
            raise NotImplementedError(f'Number of future frames {self.n_future}.')

        prediction_time = time.time()

        output['perception_time'] = perception_time - start_time
        output['prediction_time'] = prediction_time - perception_time
        output['total_time'] = output['perception_time'] + output['prediction_time']
        
        output = {**output, **bev_output}

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = unpack_sequence_dim(x, b, s)
        return x
