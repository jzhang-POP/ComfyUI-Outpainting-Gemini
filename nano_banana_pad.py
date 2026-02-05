"""
ComfyUI custom node to compute padding for Nano Banana Pro target dimensions.
Maps standard API aspect ratios to actual output dimensions.
"""

# Mapping: API aspect_ratio -> actual output (W, H) per resolution
# API accepts: 1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
DIMENSION_MAP = {
    # api_ratio: {resolution: (W, H)}
    "1:1": {
        "1K": (1024, 1024),
        "2K": (2048, 2048),
        "4K": (4096, 4096),
    },
    "5:4": {
        "1K": (1152, 928),
        "2K": (2304, 1856),
        "4K": (4608, 3712),
    },
    "4:5": {
        "1K": (928, 1152),
        "2K": (1856, 2304),
        "4K": (3712, 4608),
    },
    "4:3": {
        "1K": (1200, 896),
        "2K": (2400, 1792),
        "4K": (4800, 3584),
    },
    "3:4": {
        "1K": (896, 1200),
        "2K": (1792, 2400),
        "4K": (3584, 4800),
    },
    "3:2": {
        "1K": (1264, 848),
        "2K": (2528, 1696),
        "4K": (5056, 3392),
    },
    "2:3": {
        "1K": (848, 1264),
        "2K": (1696, 2528),
        "4K": (3392, 5056),
    },
    "16:9": {
        "1K": (1376, 768),
        "2K": (2752, 1536),
        "4K": (5504, 3072),
    },
    "9:16": {
        "1K": (768, 1376),
        "2K": (1536, 2752),
        "4K": (3072, 5504),
    },
    "21:9": {
        "1K": (1584, 672),
        "2K": (3168, 1344),
        "4K": (6336, 2688),
    },
}

VALID_ASPECT_RATIOS = ["auto", "1:1", "5:4", "4:5", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"]
VALID_RESOLUTIONS = ["auto", "1K", "2K", "4K"]


def get_all_dimensions():
    """Flatten DIMENSION_MAP into list of (W, H, api_ratio, resolution)."""
    dims = []
    for api_ratio, res_map in DIMENSION_MAP.items():
        for res, (w, h) in res_map.items():
            dims.append((w, h, api_ratio, res))
    return dims


def get_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """Get (W, H) for a specific aspect_ratio and resolution."""
    if aspect_ratio not in DIMENSION_MAP:
        raise ValueError(f"Unknown aspect_ratio: {aspect_ratio}")
    if resolution not in DIMENSION_MAP[aspect_ratio]:
        raise ValueError(f"Unknown resolution: {resolution}")
    return DIMENSION_MAP[aspect_ratio][resolution]


def find_best_fit(W: int, H: int, must_grow: bool = True) -> tuple[str, str]:
    """Find smallest supported dimension that contains the image."""
    all_dims = get_all_dimensions()
    
    if must_grow:
        candidates = [(w, h, ar, res) for w, h, ar, res in all_dims 
                      if w >= W and h >= H and (w > W or h > H)]
    else:
        candidates = [(w, h, ar, res) for w, h, ar, res in all_dims 
                      if w >= W and h >= H]
    
    if not candidates:
        raise ValueError(
            f"Image {W}x{H} exceeds all supported sizes. "
            f"Maximum supported is 6336x2688 (21:9 @ 4K) or 3072x5504 (9:16 @ 4K)."
        )
    
    # Sort by total pixels (smallest first)
    candidates.sort(key=lambda x: x[0] * x[1])
    _, _, best_ar, best_res = candidates[0]
    
    return best_ar, best_res


class NanaBananaPadCalculator:
    """Compute padding to reach a Nano Banana Pro dimension."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": ("STRING", {"default": "auto"}),
                "resolution": ("STRING", {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("pad_left", "pad_right", "pad_top", "pad_bottom", "target_w", "target_h", "aspect_ratio", "resolution")
    FUNCTION = "calculate"
    CATEGORY = "image/padding"

    def calculate(self, image, aspect_ratio: str, resolution: str):
        # image shape: (batch, H, W, C)
        _, H, W, _ = image.shape
        
        # Validate inputs
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio '{aspect_ratio}'. Valid: {VALID_ASPECT_RATIOS}")
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(f"Invalid resolution '{resolution}'. Valid: {VALID_RESOLUTIONS}")
        
        # Handle auto modes
        if aspect_ratio == "auto" and resolution == "auto":
            aspect_ratio, resolution = find_best_fit(W, H, must_grow=True)
        
        elif aspect_ratio == "auto":
            # Fixed resolution, find best aspect ratio (must be strictly larger)
            candidates = []
            for api_ratio, res_map in DIMENSION_MAP.items():
                if resolution in res_map:
                    tw, th = res_map[resolution]
                    if tw >= W and th >= H and (tw > W or th > H):
                        candidates.append((tw * th, api_ratio))
            
            if not candidates:
                raise ValueError(
                    f"Image {W}x{H} too large for {resolution}. Choose higher resolution."
                )
            candidates.sort()
            aspect_ratio = candidates[0][1]
        
        elif resolution == "auto":
            # Fixed aspect ratio, find smallest resolution that is strictly larger
            for res in ["1K", "2K", "4K"]:
                tw, th = get_dimensions(aspect_ratio, res)
                if tw >= W and th >= H and (tw > W or th > H):
                    resolution = res
                    break
            else:
                raise ValueError(
                    f"Image {W}x{H} too large for {aspect_ratio} at any resolution."
                )
        
        # Get final target dimensions
        tw, th = get_dimensions(aspect_ratio, resolution)
        
        if tw < W or th < H:
            raise ValueError(
                f"Image {W}x{H} is larger than target {tw}x{th} ({aspect_ratio} @ {resolution}). "
                f"Choose a higher resolution or different aspect ratio."
            )
        
        pad_h = tw - W
        pad_v = th - H
        
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left
        pad_top = pad_v // 2
        pad_bottom = pad_v - pad_top
        
        return (pad_left, pad_right, pad_top, pad_bottom, tw, th, aspect_ratio, resolution)


NODE_CLASS_MAPPINGS = {
    "NanaBananaPadCalculator": NanaBananaPadCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanaBananaPadCalculator": "Nano Banana Pad Calculator"
}