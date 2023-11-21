# This is my version of the demo

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    sam_checkpoint = "./pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    # hq_token_only: False means use hq output to correct SAM output.
    #                True means use hq output only.
    #                Default: False
    hq_token_only = False
    # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
    # For images contain single object, we suggest to set hq_token_only = True
    # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

    image = cv2.imread('input_images/example-7.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)


    #input_box = np.array([[25,25,350,500]])
    #input_point, input_label = None, None
    #hq_token_only = True
    input_point = np.array([[25,25],[100,100],[240,250]])
    input_point = np.array([[25,25]])
    input_label = np.ones(input_point.shape[0])
    input_box = None
    hq_token_only = True

    batch_box = False if input_box is None else len(input_box)>1
    result_path = 'lee_hq_sam_result/'
    os.makedirs(result_path, exist_ok=True)

    if not batch_box:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box = input_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        show_res(masks,scores,input_point, input_label, input_box, result_path + 'example-7', image)

    else:
        masks, scores, logits = predictor.predict_torch(
            point_coords=input_point,
            point_labels=input_label,
            boxes=transformed_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        masks = masks.squeeze(1).cpu().numpy()
        scores = scores.squeeze(1).cpu().numpy()
        input_box = input_box.cpu().numpy()
        show_res_multi(masks, scores, input_point, input_label, input_box, result_path + 'example-7', image)

"""
        if i==0: # bar lady
            input_box = np.array([[155,32,569,800]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==1: # bracelet
            input_box = np.array([[45,99,914,780]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==2: # drink
            input_point = np.array([[167,285]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
            hq_token_only = True
        elif i==3: # flowers lady
            input_point = np.array([[310,130]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
            hq_token_only = True
        elif i==4: # old lady robot
            input_point = np.array([[75,120]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
            hq_token_only = True
        elif i==5: # hi res flowers
            input_point = np.array([[1148,1567]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==6: # johnny haynes
            input_box = np.array([[202,0,988,922]])
            input_point, input_label = None, None
            hq_token_only = True


        if i==0:
            input_box = np.array([[4,13,1007,1023]])
            input_point, input_label = None, None
        elif i==1:
            input_box = np.array([[306, 132, 925, 893]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==2:
            input_point = np.array([[495,518],[217,140]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
            hq_token_only = True
        elif i==3:
            input_point = np.array([[221,482],[498,633],[750,379]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==4:
            input_box = np.array([[64,76,940,919]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==5:
            input_point = np.array([[373,363], [452, 575]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==6:
            input_box = np.array([[181, 196, 757, 495]])
            input_point, input_label = None, None
        elif i==7:
            # multi box input
            input_box = torch.tensor([[45,260,515,470], [310,228,424,296]],device=predictor.device)
            transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
            input_point, input_label = None, None
"""