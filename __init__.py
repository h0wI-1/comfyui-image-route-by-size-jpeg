import os
import numpy as np
import torch
from PIL import Image
import folder_paths


class ImageRouteBySize:
    """
    Routes an image to one of two outputs based on width.
      - large:        the image when width > max_width, else a 1x1 white placeholder
      - small:        the image when width <= max_width, else a 1x1 white placeholder
      - needs_resize: INT  1 = large path active, 0 = small path active

    Use ImagePickByFlag downstream to merge the two paths back into one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":     ("IMAGE",),
                "max_width": ("INT", {"default": 1248, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES  = ("large", "small", "needs_resize")
    FUNCTION      = "route"
    CATEGORY      = "image/routing"

    def route(self, image, max_width):
        w = image.shape[2]
        placeholder = torch.ones(1, 1, 1, 3, dtype=image.dtype, device=image.device)
        if w > max_width:
            return (image, placeholder, 1)
        else:
            return (placeholder, image, 0)


class ImagePickByFlag:
    """
    Merges the two outputs of ImageRouteBySize back into one image.
      needs_resize = 1  ->  returns image_a  (came from the resize / large path)
      needs_resize = 0  ->  returns image_b  (came from the direct / small path)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a":      ("IMAGE",),
                "image_b":      ("IMAGE",),
                "needs_resize": ("INT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "pick"
    CATEGORY     = "image/routing"

    def pick(self, image_a, image_b, needs_resize):
        return (image_a if needs_resize == 1 else image_b,)


class SaveImageJPEG:
    """
    Saves images as JPEG to the ComfyUI output folder.
    Returns the saved filenames so they appear in the history viewer.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output"}),
                "quality":         ("INT",    {"default": 95, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION     = "save_jpeg"
    OUTPUT_NODE  = True
    CATEGORY     = "image/routing"

    def save_jpeg(self, images, filename_prefix, quality):
        results = []
        for image in images:
            img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, "RGB")

            counter = 1
            while True:
                filename = f"{filename_prefix}_{counter:05d}_.jpg"
                if not os.path.exists(os.path.join(self.output_dir, filename)):
                    break
                counter += 1

            pil_img.save(os.path.join(self.output_dir, filename), "JPEG", quality=quality)
            results.append({"filename": filename, "subfolder": "", "type": self.type})

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "ImageRouteBySize": ImageRouteBySize,
    "ImagePickByFlag":  ImagePickByFlag,
    "SaveImageJPEG":    SaveImageJPEG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRouteBySize": "Image Route By Size",
    "ImagePickByFlag":  "Image Pick By Flag",
    "SaveImageJPEG":    "Save Image (JPEG)",
}
