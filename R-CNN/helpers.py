import torch as tc
import cv2
import debug
from selective_search import selective_search, box_filter
from tqdm import tqdm


def warp_pad(
    image: tc.Tensor,
    box: tuple[int, int, int, int],
    output_h: int = 227,
    output_w: int = 227,
    padding: int = 16,
) -> tc.Tensor:
    """Warp a region of an image specified by a bounding box + padding to a specified dimension

    Args:
        image: tensor with shape (C, H, W)
        box: description of boundary box in form (x1, y1, x2, y2)
        output_h: transformed image height
        output_w: transformed image width
        padding: how much more image to include outside of the given boundary box (basically extend boundary box by this amt)

    Returns:
        A transformed image (tensor) with shape (C, output_h, output_w)
    """
    image = cv2.copyMakeBorder(
        image.permute(1, 2, 0).numpy(),
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    image = image[
        box[1] : box[3] + 2 * padding, box[0] : box[2] + 2 * padding, :
    ]  # crop out region with the padding

    image = cv2.resize(image, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
    image = tc.from_numpy(image).permute(2, 0, 1)

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
