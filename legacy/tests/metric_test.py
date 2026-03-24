import sys
import torch
sys.path.append("/Users/boting/li_4yp/")

import pytest
from li_4yp.training import tolerance_accuracy


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]]), torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]]), [1.0, 1.0, 1.0, 1.0, 1.0]),  # batch, perfect
        (torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]]), torch.tensor([[2, 3, 4, 5], [4, 5, 6, 7]]), [0.0, 1.0, 0.0, 0.0, 0.0]),  # batch, off by 1
        (torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]]), torch.tensor([[2, 2, 4, 4], [3, 4, 5, 6]]), [0.5, 1.0, 1.0, 0.5, 1.0]),  # batch, mixed
    ]
)

def test_tolerance_accuracy(y_pred, y_true, expected):
    assert tolerance_accuracy(y_pred, y_true) == pytest.approx(expected)
