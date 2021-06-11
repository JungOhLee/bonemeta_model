# Augmentation
import random
import math
import torch
import torch.nn.functional as F

def get_transform(image, label, augmentation_dict={'flip': True, 'offset': 0.1, 'scale':0.2, 'rotate':True,'noise': 0.1}):
    # image, label = image_label_tuple
    image, label = torch.tensor(image,dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)
    label = label.unsqueeze(0).unsqueeze(0)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float


    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            image.size(),
            align_corners=False,
        )

    affine_t_label = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            label.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample(
            image,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    augmented_label = F.grid_sample(
            label,
            affine_t_label,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0].squeeze(0), augmented_label[0].squeeze(0).to(torch.bool)