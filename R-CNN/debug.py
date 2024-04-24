from selective_search import selective_search, box_filter
from typing import Union
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import cv2


def show(image: tc.Tensor) -> None:
    """Shows image given matrix of size (C, H, W)"""

    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def matrix_verbose(matrix: Union[np.ndarray, tc.Tensor]) -> None:
    """Print matrix, shape, and type"""

    assert type(matrix) == np.ndarray or type(matrix) == tc.Tensor
    print(
        matrix,
        f"\n The matrix has shape {matrix.shape} and type {matrix.dtype}. The minimum value is {matrix.min()} and the maximum value is {matrix.max()}",
    )


def show_selective_search(
    image: tc.Tensor, mode: str = "fast", random_sort: bool = False, top_n: int = 2000
) -> None:
    """Show image with top 2000 bounding boxes generated from selective search [Uijlings et al. 2013]

    Args:
        image: tensor with shape (C, H, W)
        mode: single, fast, or quality
        random_sort: refer to selective-search docs
        top_n: upper bound of boxes

    Returns:
        None, opens up matplotlib window

    """
    boxes = selective_search(image.permute(1, 2, 0), mode=mode, random_sort=random_sort)
    filter = box_filter(boxes, topN=top_n)

    for x1, y1, x2, y2 in filter:
        cv2.rectangle(
            image.permute(1, 2, 0).numpy(), (x1, y1), (x2, y2), (255, 0, 0), 1
        )
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
