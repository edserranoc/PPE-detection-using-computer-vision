import numpy as np
import pytest

from ppe.inference.preprocess import ImageReadError, preprocess, read_image


def test_letterbox_adds_borders_and_keeps_aspect():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, info = preprocess(img, 640, "letterbox")
    assert out.shape == (640, 640, 3)
    assert info.scale_x == info.scale_y == 0.5
    assert info.pad_x == 0 and info.pad_y == 140
    assert (out[0, 0] == 114).all()                    # borde gris


def test_letterbox_box_roundtrip_to_original_coordinates():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, info = preprocess(img, 640, "letterbox")
    # caja original (400, 200, 600, 500) -> espacio del modelo
    boxed = (400 * 0.5 + info.pad_x, 200 * 0.5 + info.pad_y, 600 * 0.5 + info.pad_x, 500 * 0.5 + info.pad_y)
    back = info.to_original(boxed)
    assert back == pytest.approx((400, 200, 600, 500))


def test_upscale_small_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    out, info = preprocess(img, 640, "letterbox")
    assert out.shape == (640, 640, 3) and info.scale_x == 2.0
    assert info.to_original((0, 0, 640, 640)) == (0.0, 0.0, 320.0, 240.0)   # recortado a los límites


def test_resize_mode_stretches():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, info = preprocess(img, 640, "resize")
    assert out.shape == (640, 640, 3)
    assert info.scale_x == 0.5 and info.scale_y == pytest.approx(640 / 720)
    assert info.to_original((0, 0, 640, 640)) == pytest.approx((0, 0, 1280, 720))


def test_none_mode_is_passthrough():
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    out, info = preprocess(img, 640, "none")
    assert out is img and info.to_original((10, 10, 20, 20)) == (10, 10, 20, 20)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10, 3), dtype=np.uint8), 640, "otro")


def test_read_image_errors(tmp_path):
    with pytest.raises(ImageReadError):
        read_image(tmp_path / "no_existe.jpg")
    (tmp_path / "vacia.jpg").write_bytes(b"")
    with pytest.raises(ImageReadError):
        read_image(tmp_path / "vacia.jpg")
    (tmp_path / "corrupta.jpg").write_bytes(b"esto no es una imagen")
    with pytest.raises(ImageReadError):
        read_image(tmp_path / "corrupta.jpg")
