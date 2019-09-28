import cv2
import random
import math
import numpy as np
import matplotlib.colors

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from PIL import Image


from chainercv.datasets import voc_semantic_segmentation_label_names as label_name
from chainercv.datasets import voc_semantic_segmentation_label_colors as label_color
from chainercv.visualizations import vis_instance_segmentation, vis_semantic_segmentation
from chainercv.visualizations import vis_bbox

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)

def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img[0], target_scale, order)

def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return [np.fliplr(m) for m in img]
        else:
            return np.fliplr(img)
    else:
        return img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images


def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container


def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)


def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride


def compress_range(arr):
    uniques = np.unique(arr)
    maximum = np.max(uniques)

    d = np.zeros(maximum+1, np.int32)
    d[uniques] = np.arange(uniques.shape[0])

    out = d[arr]
    return out - np.min(out)


def get_voc_palette(num_classes=21):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette


def save_instance_image(image, mask, label=None, label_names=None, filename=None):
    vis_instance_segmentation(
            image, mask, label, label_names=label_names
        )
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_semantic_image(image, label, label_names=label_name, label_colors=label_color, filename=None):
    ax, legend_handles = vis_semantic_segmentation(
            image, label, label_names=label_names, label_colors=label_colors, alpha=0.6, all_label_names_in_legend=False
        )

    # ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)

    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_bbox(img, bbox, filename=None, label=None, score=None, label_names=None, instance_colors=None, alpha=1.0, linewidth=3.0, sort_by_score=True, ax=None):

    vis_bbox(img, bbox, label=label, score=score, label_names=label_names, instance_colors=instance_colors,
                 alpha=alpha, linewidth=linewidth, sort_by_score=sort_by_score, ax=ax)

    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0. But, we transpose [3,H,W] to [H,W,3]
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    mask = np.transpose(mask, (1,2,0)) # -> 1.[3,H,W] -> [H,W,3]に変更する
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compose_cam_image(cam=None, image=None, cam_maxed=False):
    """
    Keyword Arguments:
        cam {numpy.ndarray} -- (H, W, C)
        image {numpy.ndarray} -- (H, W, C)
        cam_maxed {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    # if not cam_maxed:
    #     cam = np.sum(cam, axis=0)
    if np.max(cam) > 1:
        cam = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    else:
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
    # jetcam = jetcam.transpose(2,0,1)
    # print("cam: ", cam.shape, "image: ", image.shape)
    cam = ((np.float32(cam) + image) / 2)
    # jetcam = jetcam.transpose(1,2,0)
    return cam



def colorize_score(score_map, exclude_zero=False, normalize=True, by_hue=False):
    import matplotlib.colors
    if by_hue:
        aranged = np.arange(score_map.shape[0]) / (score_map.shape[0])
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1)
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

        test = rgb_color[np.argmax(score_map, axis=0)]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test

        if normalize:
            return test / (np.max(test) + 1e-5)
        else:
            return test

    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                     (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                     (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                     (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]

        test = VOC_color[np.argmax(score_map, axis=0)%22]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test
        if normalize:
            test /= np.max(test) + 1e-5

        return test


def colorize_displacement(disp):

    a = (np.arctan2(-disp[0], -disp[1]) / math.pi + 1) / 2

    r = np.sqrt(disp[0] ** 2 + disp[1] ** 2)
    s = r / np.max(r)
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1)
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

    return rgb_color, hsv_color


def colorize_label(label_map, normalize=True, by_hue=True, exclude_zero=False, outline=False):

    label_map = label_map.astype(np.uint8)

    if by_hue:
        import matplotlib.colors
        sz = np.max(label_map)
        aranged = np.arange(sz) / sz
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1)
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)
        rgb_color = np.concatenate([np.zeros((1, 3)), rgb_color], axis=0)

        test = rgb_color[label_map]
    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                              (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]
        test = VOC_color[label_map]
        if normalize:
            test /= np.max(test)

    if outline:
        edge = np.greater(np.sum(np.abs(test[:-1, :-1] - test[1:, :-1]), axis=-1) + np.sum(np.abs(test[:-1, :-1] - test[:-1, 1:]), axis=-1), 0)
        edge1 = np.pad(edge, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        edge2 = np.pad(edge, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        edge = np.repeat(np.expand_dims(np.maximum(edge1, edge2), -1), 3, axis=-1)

        test = np.maximum(test, edge)
    return test