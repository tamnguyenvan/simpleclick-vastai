import base64

import pytest

from app.errors import ApiError
from app.validation import decode_base64_image, sample_points, validate_points


def test_data_url_is_decoded_and_size_is_enforced():
    encoded = base64.b64encode(b"image-bytes").decode()
    assert decode_base64_image(f"data:image/png;base64,{encoded}", 100) == b"image-bytes"

    with pytest.raises(ApiError) as error:
        decode_base64_image(encoded, 2)
    assert error.value.code == "image_too_large"
    assert error.value.status_code == 413


def test_point_sampling_preserves_stroke_endpoints():
    points = [[float(index), float(index)] for index in range(100)]
    sampled = sample_points(points, 24)
    assert len(sampled) == 24
    assert sampled[0] == [0, 0]
    assert sampled[-1] == [99, 99]


def test_points_are_bounded_and_rounded_for_model_coordinates():
    assert validate_points([[1.6, 2.4]], 3, 4, "positive_points", 24, 10) == [[2, 2]]

    with pytest.raises(ApiError) as error:
        validate_points([[3, 0]], 3, 4, "positive_points", 24, 10)
    assert error.value.code == "point_out_of_bounds"

