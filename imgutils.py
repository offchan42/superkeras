"""Utilities used for manipulating images"""

import cv2 as cv
import numpy as np


class Rect:
    """Bounding rect that support floating point values. Designed to be chained
    with multiple methods. Each method creates a new immutable Rect object."""

    def __init__(self, coordinates=None, xywh=None, cxcywh=None, xyxy=None):
        """Create Rect instance with one of the way we provide.
        # Args
            coordinates: Array of shape (N, 2) containing X and Y coordinates
            xywh: A tuple (x,y,w,h) where (x,y) is the top-left of the rect
            cxcywh: A tuple (cx,cy,w,h) where (cx,cy) is the center of the rect
            xyxy: A tuple (x1,y1,x2,y2) where (x1,y1) is the top-left of the rect
                and (x2, y2) is the bottom-right.
        """
        cnt = 0
        cnt += coordinates is not None
        cnt += xywh is not None
        cnt += cxcywh is not None
        if cnt != 1:
            raise ValueError("Please set exactly one of the arguments.")
        if coordinates is not None:
            self.xywh = bounding_rect(coordinates)
        if xywh is not None:
            self.xywh = xywh
        if cxcywh is not None:
            cx, cy, w, h = cxcywh
            self.xywh = cx - w / 2, cy - h / 2, w, h
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            self.xywh = x1, y1, x2 - x1, y2 - y1

    @property
    def xyxy_int(self):
        """Used for slicing the image (x, y, x+w, y+h) or drawing rect."""
        # reminder: we don't convert self.xyxy to int to prevent off-by-one error
        # for width and height (causing by rounding error)
        x, y, w, h = self.xywh_int
        return x, y, x + w, y + h

    @property
    def xywh_int(self):
        x, y, w, h = self.xywh
        x, y, w, h = int(x), int(y), int(round(w)), int(round(h))
        return x, y, w, h

    @property
    def xyxy(self):
        x, y, w, h = self.xywh
        return x, y, x + w, y + h

    @property
    def cxcywh(self):
        x, y, w, h = self.xywh
        cx, cy = x + w / 2, y + h / 2
        return cx, cy, w, h

    def square(self):
        """Return a square Rect with side length = max(w, h)"""
        cx, cy, w, h = self.cxcywh
        w = h = max(w, h)
        return Rect(cxcywh=(cx, cy, w, h))

    def scale(self, scale):
        """Return a scaled Rect (scale=1 will do nothing).
        `scale` can be a tuple of (w, h) for scaling width and height unequally"""
        cx, cy, w, h = self.cxcywh
        if isinstance(scale, tuple):
            w *= scale[0]
            h *= scale[1]
        else:
            w *= scale
            h *= scale
        return Rect(cxcywh=(cx, cy, w, h))

    def translate(self, sx, sy):
        """Move the rect (sx, sy) units"""
        x, y, w, h = self.xywh
        return Rect(xywh=(x + sx, y + sy, w, h))

    def draw(self, img, color, thick=1):
        """Draw rectangle onto the image inplace"""
        x1, y1, x2, y2 = self.xyxy_int
        cv.rectangle(img, (x1, y1), (x2, y2), color, thickness=thick)

    def slice_coords(self, coords):
        """Return coords - np.float32([topleft_x, topleft_y])"""
        x, y, w, h = self.xywh_int
        return coords - np.float32([x, y])

    def slice(self, img, coords=None, zero_padding=True):
        """Slice the image with this rect, and return the sliced part as a new image.
        The output will be guaranteed to be a copy if `zero_padding` is True.
        If `coords` for `img` is provided, we will convert it to be relative to
        the sliced image and return it along with the new image. `coords` must
        contain (x, y) values, not (y, x) values.
        """
        assert len(img.shape) in [2, 3]
        if zero_padding:
            x, y, w, h = self.xywh_int
            pos = [y, x]
            crop_shape = [h, w]
            if len(img.shape) == 3:  # crop color dimension fully
                pos.append(0)
                crop_shape.append(img.shape[-1])
            crop = np.zeros(crop_shape, dtype=img.dtype)
            fill_crop(img, pos, crop)
            if coords is not None:
                return crop, coords - np.float32([x, y])
            return crop
        else:
            x1, y1, x2, y2 = self.xyxy_int
            crop = img[y1:y2, x1:x2]
            if coords is not None:
                return crop, coords - np.float32([x1, y1])
            return crop

    def represent(self):
        return dict(
            xywh=self.xywh,
            xyxy=self.xyxy,
            cxcywh=self.cxcywh,
            xywh_int=self.xywh_int,
            xyxy_int=self.xyxy_int,
        )

    def __repr__(self):
        return repr(self.represent())

    def __str__(self):
        return str(self.represent())


def bounding_rect(coordinates):
    """Find bounding rect of x,y coordinates with floating point values."""
    x, y = np.min(coordinates, axis=0)
    x2, y2 = np.max(coordinates, axis=0)
    w, h = x2 - x, y2 - y
    return x, y, w, h


def fill_crop(img, pos, crop):
    """
    Fills `crop` with values from `img` at `pos`,
    while accounting for the crop being off the edge of `img`.
    *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
    See example at: https://stackoverflow.com/a/55391329/2593810

    # Args
        img: The image to crop from
        pos: List of starting indices. E.g. [-2, 1] for 2 dimensional image
        crop: Output array to fill. E.g. np.zeros([3, 4], dtype=np.uint8)
    """
    img_shape, pos, crop_shape = (
        np.array(img.shape),
        np.array(pos),
        np.array(crop.shape),
    )
    end = pos + crop_shape
    # Calculate crop slice positions
    crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    pos = np.clip(pos, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = (slice(low, high) for low, high in zip(pos, end))
    crop[tuple(crop_slices)] = img[tuple(img_slices)]
