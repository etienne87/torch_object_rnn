import numpy as np
import cv2


def boxarray_to_boxes(boxarray):
    boxes = []
    for i in range(len(boxarray)):
        x1, y1, x2, y2, c = tuple(boxarray[i])
        pt1 = (x1,y1)
        pt2 = (x2,y2)
        boxes.append(('',c,pt1,pt2,None,None,None))
    return boxes

#UI Utils
def draw_bboxes(img, bboxes):
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)

    colors = [tuple(*item) for item in colors.tolist()]
    for bbox in bboxes:
        class_name, class_id, pt1, pt2, _, _, _ = bbox
        center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        color = colors[class_id * 60]
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return img

def filter_outliers(g, num_std=2):
    grange = num_std*g.std()
    gimg_min = g.mean() - grange
    gimg_max = g.mean() + grange
    g_normed = np.minimum(np.maximum(g,gimg_min),gimg_max) #clamp
    return g_normed


def make_single_channel_display(img, low=None, high=None):
    if low is None:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img = ((img+1)/2*255).astype(np.uint8)
    img = (img.astype(np.uint8))[..., None].repeat(3, 2)
    return img