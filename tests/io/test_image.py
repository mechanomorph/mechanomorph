import numpy as np
from skimage.io import imsave
import pytest

from mechanomorph.io import read_lazy_tiff


@pytest.mark.filterwarnings("ignore:ignoring keyword argument")
def test_read_lazy_tiff(tmp_path):
    """Test reading a tiff."""
    image_shape = (5, 10, 10, 10)
    im = np.ones(image_shape)
    image_path = tmp_path / "test.tif"
    imsave(image_path, im, check_contrast=False)

    loaded_im = read_lazy_tiff(image_path)
    assert loaded_im.shape == image_shape
