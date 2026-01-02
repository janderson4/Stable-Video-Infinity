import numpy as np

def visualize_video_tensor(video, save_path=None, fps=15):
    """
    Minimal stub for visualize_video_tensor used by WanVideoSviProPipeline.

    Args:
        video: torch.Tensor or np.ndarray of shape
               (T, C, H, W) or (C, T, H, W) or (T, H, W, C).
        save_path: Optional path to write a gif/mp4. If None, this is a no-op.
        fps: Frames per second if saving.

    This implementation is intentionally lightweight – for our streaming setup
    we don't rely on it, but SVI imports it at module import time.
    """
    # We make it a safe no-op if save_path is None.
    if save_path is None:
        return

    try:
        import torch
        import imageio
    except ImportError:
        # If imageio/torch aren't available (shouldn't happen here), just bail.
        return

    if isinstance(video, torch.Tensor):
        v = video.detach().cpu()
        if v.dim() == 4:
            # Try to normalize to (T, H, W, C)
            if v.shape[1] in (1, 3):   # (T, C, H, W)
                v = v.permute(0, 2, 3, 1)  # (T, H, W, C)
            elif v.shape[0] in (1, 3): # (C, T, H, W)
                v = v.permute(1, 2, 3, 0)  # (T, H, W, C)
        frames = v
    else:
        frames = np.array(video)

    # Normalize to uint8
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 1)
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(save_path, frames, fps=fps)
