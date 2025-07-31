import torch
import cv2
import numpy as np
from pathlib import Path


from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from vitpose_model import ViTPoseModel


from detectron2.config import LazyConfig
import hamer
cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cpm = ViTPoseModel(device)
img_path = "hand_recon/pringles_hand/23263780.png"


img_cv2 = cv2.imread(str(img_path))
save_img_cv2 = cv2.imread("hand_recon/pringles_out/23263780_all_kpt.jpg")

det_out = detector(img_cv2)
img = img_cv2.copy()[:, :, ::-1]

det_instances = det_out['instances']
valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
pred_scores=det_instances.scores[valid_idx].cpu().numpy()

vitposes_out = cpm.predict_pose(
    img,
    [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
)

bboxes = []
is_right = []
keyps = []
keypoint_scores = []

# Use hands based on hand keypoint detections
for vitposes in vitposes_out:
    left_hand_keyp = vitposes['keypoints'][-42:-21]
    right_hand_keyp = vitposes['keypoints'][-21:]
    # Rejecting not confident detections
    keyp = left_hand_keyp
    valid = keyp[:,2] > 0.5
    if sum(valid) > 3:
        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
        bboxes.append(bbox)
        keyps.append(keyp[:, :2])
        keypoint_scores.append(left_hand_keyp[:,2])
        is_right.append(0)
    keyp = right_hand_keyp
    valid = keyp[:,2] > 0.5
    if sum(valid) > 3:
        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
        bboxes.append(bbox)
        keyps.append(keyp[:, :2])
        keypoint_scores.append(right_hand_keyp[:,2])
        is_right.append(1)

# if len(bboxes) == 0:
#     continue

boxes = np.stack(bboxes)
right = np.stack(is_right)
keyp_scores = np.array(keypoint_scores)

print(right, keyps, keyp_scores)

for keyp in keyps:
    for kp in keyp:
        x, y = int(kp[0]), int(kp[1])
            #if 0 <= x < img_w and 0 <= y < img_h:
        cv2.circle(save_img_cv2, (x, y), 2, [0.0, 0.0, 256])

cv2.imwrite('image.png', save_img_cv2)