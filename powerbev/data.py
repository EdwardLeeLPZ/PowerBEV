# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

import os
from operator import rshift

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from cv2 import fastNlMeansDenoising
from lyft_dataset_sdk.lyftdataset import LyftDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from powerbev.utils.geometry import (calculate_birds_eye_view_parameters,
                                     convert_egopose_to_matrix_numpy,
                                     invert_matrix_egopose_numpy, mat2pose_vec,
                                     pose_vec2mat, resize_and_crop_image,
                                     update_intrinsics)
from powerbev.utils.instance import \
    convert_instance_mask_to_center_and_offset_label
from powerbev.utils.lyft_splits import TRAIN_LYFT_INDICES, VAL_LYFT_INDICES
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


class FuturePredictionDataset(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.is_train = is_train
        self.cfg = cfg

        self.is_lyft = isinstance(nusc, LyftDataset)

        if self.is_lyft:
            self.dataroot = self.nusc.data_path
        else:
            self.dataroot = self.nusc.dataroot

        self.mode = 'train' if self.is_train else 'val'

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES

        self.scenes = self.get_scenes()
        self.ixes = self.get_samples()
        self.indices = self.get_indices()

        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

    def get_scenes(self):
        """
        Obtain the list of scenes names in the given split.
        """
        if self.is_lyft:
            scenes = [row['name'] for row in self.nusc.scene]

            # Split in train/val
            indices = TRAIN_LYFT_INDICES if self.is_train else VAL_LYFT_INDICES
            scenes = [scenes[i] for i in indices]
        else:
            # filter by scene split
            split = {'v1.0-trainval': {True: 'train', False: 'val'},
                     'v1.0-mini': {True: 'mini_train', False: 'mini_val'},}[
                self.nusc.version
            ][self.is_train]

            scenes = create_splits_scenes()[split]

        return scenes

    def get_samples(self):
        """
        Find and sort the samples in the given split by scene.
        """
        samples = [sample for sample in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [sample for sample in samples if self.nusc.get('scene', sample['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        """
        Group the sample indices by sequence length.
        """
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        """
        Determine the parameters for preprocessing of the input image, e.g. image size after scaling, cropping area.
        """
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, rec):
        """
        Obtain the input image as well as the intrinsics and extrinsics parameters of the corresponding camera.

        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        cameras = self.cfg.IMAGE.NAMES

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )

        return images, intrinsics, extrinsics

    def _get_top_lidar_pose(self, rec):
        """
        Obtain the vehicle attitude at the current moment.
        """
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def record_instance(self, rec, instance_map):
        """
        Record information about each visible instance in the sequence and assign a unique ID to it.
        """
        translation, rotation = self._get_top_lidar_pose(rec)
        self.egopose_list.append([translation, rotation])

        for annotation_token in rec['anns']:
            
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if not self.is_lyft:
                # NuScenes filter
                # Filter out all non vehicle instances
                if 'vehicle' not in annotation['category_name']:
                    continue
                # Filter out invisible vehicles
                if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in self.visible_instance_set:
                    continue
                # Filter out vehicles that have not been seen in the past
                if self.counter >= self.cfg.TIME_RECEPTIVE_FIELD and annotation['instance_token'] not in self.visible_instance_set:
                    continue
                self.visible_instance_set.add(annotation['instance_token'])
            else:
                # Lyft filter
                # Filter out all non vehicle instances
                if annotation['category_name'] not in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
                    continue

            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]

            if not self.is_lyft:
                instance_attribute = int(annotation['visibility_token'])
            else:
                instance_attribute = 0
            
            if annotation['instance_token'] not in self.instance_dict:
                # For the first occurrence of an instance
                self.instance_dict[annotation['instance_token']] = {
                    'timestep': [self.counter],
                    'translation': [annotation['translation']],
                    'rotation': [annotation['rotation']],
                    'size': annotation['size'],
                    'instance_id': instance_id,
                    'attribute_label': [instance_attribute],
                }
            else:
                # For the instance that have appeared before
                self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
                self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map
    
    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments.
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True

    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise.
        """
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.sequence_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer-1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
                pointer += 1
                continue
            
            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
                instance['translation'][pointer] = instance['translation'][pointer-1]
                instance['rotation'][pointer] = instance['rotation'][pointer-1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
            pointer += 1
        
        return instance

    @staticmethod
    def generate_flow(flow, instance_img, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation.
        """
        _, h, w = instance_img.shape
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))
        grid = torch.stack((x, y), dim=0)

        # Set the first frame
        instance_mask = (instance_img[0] == instance_id)
        flow[0, 1, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[0, 0, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (instance_img[timestep] == instance_id)
            prev_instance_mask = (instance_img[timestep-1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            # Centripetal backward flow is defined as displacement vector from each foreground pixel at time t to the object center of the associated instance identity at time t−1
            flow[timestep, 1, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
            flow[timestep, 0, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        return flow
    
    def get_flow_label(self, instance_img, instance_map, ignore_index=255):
        """
        Generate the global map of the flow ground truth.
        """
        seq_len, h, w = instance_img.shape
        flow = ignore_index * torch.ones(seq_len, 2, h, w)

        for token, instance in self.instance_dict.items():
            flow = self.generate_flow(flow, instance_img, instance, instance_map[token])
        return flow

    def _get_poly_region_in_image(self, instance_annotation, present_egopose):
        """
        Obtain the bounding box polygon of the instance.
        """
        present_ego_translation, present_ego_rotation = present_egopose

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)
        pts = box.bottom_corners()[:2].T

        if self.cfg.LIFT.X_BOUND[0] <= pts.min(axis=0)[0] and pts.max(axis=0)[0] <= self.cfg.LIFT.X_BOUND[1] and self.cfg.LIFT.Y_BOUND[0] <= pts.min(axis=0)[1] and pts.max(axis=0)[1] <= self.cfg.LIFT.Y_BOUND[1]:
            pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            z = box.bottom_corners()[2, 0]
            return pts, z
        else:
            return None, None

    def get_label(self):
        """
        Generate labels for semantic segmentation, instance segmentation, z position, attribute from the raw data of nuScenes.
        """
        timestep = self.counter
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        z_position = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        attribute_label = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        
        for instance_token, instance_annotation in self.instance_dict.items():
            if timestep not in instance_annotation['timestep']:
                continue
            pointer = instance_annotation['timestep'].index(timestep)
            annotation = {
                'translation': instance_annotation['translation'][pointer],
                'rotation': instance_annotation['rotation'][pointer],
                'size': instance_annotation['size'],
            }
            poly_region, z = self._get_poly_region_in_image(annotation, self.egopose_list[self.cfg.TIME_RECEPTIVE_FIELD - 1]) 
            if isinstance(poly_region, np.ndarray):
                if self.counter >= self.cfg.TIME_RECEPTIVE_FIELD and instance_token not in self.visible_instance_set:
                    continue
                self.visible_instance_set.add(instance_token)

                cv2.fillPoly(instance, [poly_region], instance_annotation['instance_id'])
                cv2.fillPoly(segmentation, [poly_region], 1.0)
                cv2.fillPoly(z_position, [poly_region], z)
                cv2.fillPoly(attribute_label, [poly_region], instance_annotation['attribute_label'][pointer]) 
        
        segmentation = torch.from_numpy(segmentation).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance).long().unsqueeze(0)
        z_position = torch.from_numpy(z_position).float().unsqueeze(0).unsqueeze(0)
        attribute_label = torch.from_numpy(attribute_label).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, z_position, attribute_label

    def get_future_egomotion(self, rec, index):
        """
        Obtain the egomotion in the corresponding sequence from the raw data of nuScenes.
        """
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1
                sample_token: List<str> (T,)
                'z_position': list_z_position,
                'attribute': list_attribute_label,

        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'future_egomotion',
                'sample_token',
                'z_position', 'attribute'
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # The visible instance must have high visibility in the time receptive field
        self.visible_instance_set = set()
        # Record all valid instance
        self.instance_dict = {}
        # Record translation and rotation of the ego-vehicle for the entire time period
        self.egopose_list = []
        # Generate input data
        for self.counter, index_t in enumerate(self.indices[index]):
            rec = self.ixes[index_t]

            images, intrinsics, extrinsics = self.get_input_data(rec)
            instance_map = self.record_instance(rec, instance_map)
            future_egomotion = self.get_future_egomotion(rec, index_t)

            data['image'].append(images)
            data['intrinsics'].append(intrinsics)
            data['extrinsics'].append(extrinsics)
            data['future_egomotion'].append(future_egomotion)
            data['sample_token'].append(rec['token'])

        # Refine the generated instance polygons    
        for token in self.instance_dict.keys():
            self.instance_dict[token] = self.refine_instance_poly(self.instance_dict[token])

        # The visible instance must have high visibility in the time receptive field
        self.visible_instance_set = set()
        # Generate instance ground truth
        for self.counter in range(self.sequence_length):
            segmentation, instance, z_position, attribute_label = self.get_label()
            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['z_position'].append(z_position)
            data['attribute'].append(attribute_label)

        for key, value in data.items():
            if key in ['sample_token', 'centerness', 'offset', 'flow']:
                continue
            data[key] = torch.cat(value, dim=0)

        # Generate centripetal backward flow ground truth from the instance ground truth
        data['flow'] = self.get_flow_label(data['instance'], instance_map, ignore_index=self.cfg.DATASET.IGNORE_INDEX)

        # If lyft need to subsample, and update future_egomotions
        if self.cfg.MODEL.SUBSAMPLE:
            for key, value in data.items():
                if key in ['future_egomotion', 'sample_token', 'centerness', 'offset', 'flow']:
                    continue
                data[key] = data[key][::2].clone()
            data['sample_token'] = data['sample_token'][::2]

            # Update future egomotions
            future_egomotions_matrix = pose_vec2mat(data['future_egomotion'])
            future_egomotion_accum = torch.zeros_like(future_egomotions_matrix)
            future_egomotion_accum[:-1] = future_egomotions_matrix[:-1] @ future_egomotions_matrix[1:]
            future_egomotion_accum = mat2pose_vec(future_egomotion_accum)
            data['future_egomotion'] = future_egomotion_accum[::2].clone()

        # Generate the ground truth of centerness and offset
        instance_centerness, instance_offset = convert_instance_mask_to_center_and_offset_label(
            data['instance'], num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['num_instances'] = torch.tensor([len(instance_map)])

        return data


def prepare_powerbev_dataloaders(cfg, return_dataset=False):
    """
    Prepare the dataloader of PowerBEV.
    """
    version = cfg.DATASET.VERSION
    train_on_training_data = True

    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    elif cfg.DATASET.NAME == 'lyft':
        # train contains 22680 samples
        # we split in 16506 6174
        dataroot = os.path.join(cfg.DATASET.DATAROOT, 'trainval')
        nusc = LyftDataset(data_path=dataroot,
                           json_path=os.path.join(dataroot, 'train_data'),
                           verbose=True)

    train_data = FuturePredictionDataset(nusc, train_on_training_data, cfg)
    val_data = FuturePredictionDataset(nusc, False, cfg)

    if cfg.DATASET.VERSION == 'mini':
        train_data.indices = train_data.indices[:10]
        val_data.indices = val_data.indices[:10]

    nworkers = cfg.N_WORKERS
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)

    if return_dataset:
        return train_loader, val_loader, train_data, val_data
    else:
        return train_loader, val_loader
