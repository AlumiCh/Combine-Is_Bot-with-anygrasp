import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Wrapper:
    def __init__(self, checkpoint_path, model_cfg, device='cuda'):
        self.device = device
        # 初始化 SAM2
        self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def segment(self, image, point_coords=None, point_labels=None, box=None):
        """
        对图像进行分割
        Args:
            image: (H, W, 3) RGB 图像
            point_coords: (N, 2) 提示点坐标 [[x, y], ...]
            point_labels: (N,) 提示点标签 (1前景, 0背景)
            box: [x1, y1, x2, y2] 提示框
        Returns:
            mask: (H, W) bool 类型的掩码
        """

        self.predictor.set_image(image)
        
        # 如果没有提示，可能需要自动分割或者中心点提示
        if point_coords is None and box is None:
            # 默认策略：假设物体在图像中心
            h, w = image.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])
            
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False
        )
        
        # 返回分数最高的 mask
        best_mask = masks[0]
        return best_mask.astype(bool)