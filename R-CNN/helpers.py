import torch as tc
import cv2
import debug
from selective_search import selective_search, box_filter
from tqdm import tqdm

class_map = {'boat': 0, 'person': 1, 'motorbike': 2, 'sheep': 3, 'horse': 4, 'diningtable': 5, 'aeroplane': 6, 'bicycle': 7, 'bus': 8, 'train': 9, 'sofa': 10, 'cat': 11, 'bird': 12, 'car': 13, 'cow': 14, 'pottedplant': 15, 'chair': 16, 'tvmonitor': 17, 'dog': 18, 'bottle': 19}

def warp_pad(
    image: tc.Tensor,
    box: tuple[int, int, int, int],
    output_h: int = 227,
    output_w: int = 227,
    padding: int = 16,
) -> tc.Tensor:
    """Warp a region of an image specified by a bounding box + padding to a specified dimension and then subtract mean from image

    Args:
        image: tensor with shape (C, H, W)
        box: description of boundary box in form (x1, y1, x2, y2)
        output_h: transformed image height
        output_w: transformed image width
        padding: how much more image to include outside of the given boundary box (basically extend boundary box by this amt)

    Returns:
        A transformed image (tensor) with shape (C, output_h, output_w)
    """
    r_mean, g_mean, b_mean = tc.mean(image.float(), dim=[1, 2])

    image = cv2.copyMakeBorder(
        image.permute(1, 2, 0).numpy(),
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(int(b_mean), int(g_mean), int(r_mean)),
    )

    image = image[
        box[1] : box[3] + 2 * padding, box[0] : box[2] + 2 * padding, :
    ]  # crop out region with the padding

    image = cv2.resize(image, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
    image = tc.from_numpy(image).permute(2, 0, 1)
    image = image - tc.Tensor([r_mean, g_mean, b_mean]).resize(3, 1, 1)
    return image


def extract_regions(
    image: tc.Tensor, mode: str = "fast", random_sort: bool = False, top_n: int = 2000
) -> list[tc.Tensor]:
    """Crop regions given selective search boundary boxes and warp with 16 pixel context padding
    as described in Appendix A

    Args:
        image: tensor with shape (C, H, W)
        mode: refer to selective-search docs
        random_sort: refer to selective-search docs
        top_n: upper bound on boxes

    Returns:
        List of resized images (tensors)
    """
    regions = []
    boxes = selective_search(image.permute(1, 2, 0), mode=mode, random_sort=random_sort)
    filter = box_filter(boxes, topN=top_n)
    for box in tqdm(filter):
        regions.append(warp_pad(image, box))

    return regions

def get_class(label: dict) -> int:
    """Returns an integer based on the image class. If multiple objects in the label, uses the first one.
    
    Args:
        label: a dictionary in the form of VOC detection .xml annotations
    
    Returns:
        Number based on a class map
    """
    return class_map[label['annotation']['object'][0]['name']]