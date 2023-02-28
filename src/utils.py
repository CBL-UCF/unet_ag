import numpy as np
from skimage.morphology import label


def postprocess_stack(seg):
    # basically look for connected components and choose the largest one, delete everything else
    # print('stack')
    if np.sum(seg) > 0:
        mask = seg != 0
        lbls = label(mask, 8)
        lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
        largest_region = np.argmax(lbls_sizes[1:]) + 1
        seg[lbls != largest_region] = 0
    return seg


def get_center_point(mask):
    no_mask = False
    if len(mask[mask != 0]):
        yp, xp = np.where(mask != 0)
        x_min = np.min(xp)
        x_max = np.max(xp)
        y_min = np.min(yp)
        y_max = np.max(yp)
    else:
        x_min = 0
        x_max = mask.shape[1] - 1
        y_min = 0
        y_max = mask.shape[0] - 1
        no_mask = True

    return (x_min + x_max) / 2, (y_min + y_max) / 2, no_mask


def get_stack_center_point(mask):
    x = list()
    y = list()

    for m in mask:
        xc, yc, no_mask = get_center_point(m)
        if not no_mask:
            x.append(xc)
            y.append(yc)

    if len(x) > 0:
        xc = np.round(np.mean(x)).astype(np.int)
        yc = np.round(np.mean(y)).astype(np.int)
    else:
        xc = mask.shape[-1] // 2
        yc = mask.shape[-2] // 2

    return xc, yc
