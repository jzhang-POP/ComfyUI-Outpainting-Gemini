"""
ComfyUI custom node to compute padding for Nano Banana Pro target dimensions.
Uses actual supported sizes from the API.
"""

# Actual supported dimensions: (W, H, aspect_ratio, resolution)
SUPPORTED_DIMENSIONS = [
    # Landscape
    (1584, 672, "33:14", "1K"),
    (3168, 1344, "33:14", "2K"),
    (6336, 2688, "33:14", "4K"),
    
    (1376, 768, "43:24", "1K"),
    (2752, 1536, "43:24", "2K"),
    (5504, 3072, "43:24", "4K"),
    
    (1264, 848, "79:53", "1K"),
    (2528, 1696, "79:53", "2K"),
    (5056, 3392, "79:53", "4K"),
    
    (1200, 896, "75:56", "1K"),
    (2400, 1792, "75:56", "2K"),
    (4800, 3584, "75:56", "4K"),
    
    (1152, 928, "36:29", "1K"),
    (2304, 1856, "36:29", "2K"),
    (4608, 3712, "36:29", "4K"),
    
    # Square
    (1024, 1024, "1:1", "1K"),
    (2048, 2048, "1:1", "2K"),
    (4096, 4096, "1:1", "4K"),
    
    # Portrait
    (928, 1152, "29:36", "1K"),
    (1856, 2304, "29:36", "2K"),
    (3712, 4608, "29:36", "4K"),
    
    (896, 1200, "56:75", "1K"),
    (1792, 2400, "56:75", "2K"),
    (3584, 4800, "56:75", "4K"),
    
    (848, 1264, "53:79", "1K"),
    (1696, 2528, "53:79", "2K"),
    (3392, 5056, "53:79", "4K"),
    
    (768, 1376, "24:43", "1K"),
    (1536, 2752, "24:43", "2K"),
    (3072, 5504, "24:43", "4K"),
]

VALID_ASPECT_RATIOS = ["auto", "33:14", "43:24", "79:53", "75:56", "36:29", "1:1", "29:36", "56:75", "53:79", "24:43"]
VALID_RESOLUTIONS = ["auto", "1K", "2K", "4K"]


def get_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """Get (W, H) for a specific aspect_ratio and resolution."""
    for w, h, ar, res in SUPPORTED_DIMENSIONS:
        if ar == aspect_ratio and res == resolution:
            return w, h
    raise ValueError(f"No supported size for {aspect_ratio} @ {resolution}")


def find_best_fit(W: int, H: int, must_grow: bool = True) -> tuple[str, str]:
    """Find smallest supported dimension that contains the image."""
    if must_grow:
        candidates = [(w, h, ar, res) for w, h, ar, res in SUPPORTED_DIMENSIONS 
                      if w >= W and h >= H and (w > W or h > H)]
    else:
        candidates = [(w, h, ar, res) for w, h, ar, res in SUPPORTED_DIMENSIONS 
                      if w >= W and h >= H]
    
    if not candidates:
        raise ValueError(
            f"Image {W}x{H} exceeds all supported sizes. "
            f"Maximum supported is 6336x2688 (33:14 @ 4K) or 3072x5504 (24:43 @ 4K)."
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
            for w, h, ar, res in SUPPORTED_DIMENSIONS:
                if res == resolution and w >= W and h >= H and (w > W or h > H):
                    candidates.append((w * h, ar))
            
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