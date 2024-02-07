import os
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .siglip.siglip_encoder import SigLipVisionTower
from .clip.clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'sig' in vision_tower.lower():
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    elif 'eva' in vision_tower.lower():
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'clip' in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
