from typing import Tuple, Optional
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import timm
 
class DINOVisualTokenizer(nn.Module):
    """
    Wraps a DINO ViT model (from timm) to extract patch tokens (visual tokens) T_vis.
 
    Input: image tensor of shape (B, 3, H, W) normalized appropriately.
    Output: T_vis with shape (B, M, D_s), where M is the number of patches, D_s is embed dim.
 
    Notes:
    - Uses forward hook on the model's final LayerNorm to grab the full token sequence after transformer blocks.
    - Slices off the class token to return only patch tokens.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224.dino", pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if not hasattr(self.model, "num_classes"):
            raise ValueError("Unexpected model type; expected a ViT from timm with DINO weights.")
        # Ensure model returns embeddings, not classifier logits
        if getattr(self.model, "num_classes", None) is not None and self.model.num_classes != 0:
            # Disable classifier head if present
            if hasattr(self.model, "reset_classifier"):
                self.model.reset_classifier(num_classes=0)
            else:
                # Fallback: replace head with identity
                if hasattr(self.model, "head"):
                    self.model.head = nn.Identity()
 
        # Determine embedding dimension D_s
        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            # Some timm models use 'num_features'
            self.embed_dim = getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise ValueError("Could not determine embed dim from DINO model.")
 
        # Preprocessing parameters from timm model config (normalization & input size)
        cfg = getattr(self.model, "default_cfg", {})
        self.input_size = cfg.get("input_size", (3, 224, 224))
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
 
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float tensor in range [0,1], shape (B, 3, H, W).
        Resizes to model input size and normalizes.
        """
        _, target_h, target_w = self.input_size
        if x.shape[-2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x = (x - self.mean) / (self.std+1e-3)
        return x
 
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns T_vis: visual patch tokens with shape (B, M, D_s).
        """
        dtype = x.dtype
        x = self.preprocess(x)

        captured = {}
 
        def hook(module, inp, out):
            # out has shape (B, 1 + M, D_s), first token is class token
            captured["tokens"] = out
        # Register hook on final norm to capture token sequence
        
        handle = self.model.norm.register_forward_hook(hook)
        res = self.model(x)  # standard forward; hook captures sequence at the end
        handle.remove()
 
        if "tokens" not in captured:
            raise RuntimeError("Failed to capture tokens via forward hook. Check model structure/version.")
 
        tokens = captured["tokens"]  # (B, 1 + M, D_s)
        T_vis = tokens[:, 1:, :]     # remove class token -> (B, M, D_s)
        return T_vis.to(dtype)
 
 
class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
 
class CrossSelfBlock(nn.Module):
    """
    One block consisting of:
    - Cross-attention: camera token queries visual tokens to update t_cam
      q = t_cam, k = v = T_vis
    - Self-attention: joint self-attn over [t_cam, T_vis] to update both
    Each attention sub-block is followed by an MLP with residual connections.
 
    Shapes:
    - T_vis: (B, M, D_s)
    - t_cam: (B, 1, D_s)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
 
        # Cross-attention (camera token attends visual tokens)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.cross_mlp = FeedForward(dim, mlp_ratio, proj_drop)
 
        # Self-attention (joint over [t_cam, T_vis])
        self.norm_joint = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.self_mlp = FeedForward(dim, mlp_ratio, proj_drop)
 
        self.drop = nn.Dropout(proj_drop)
 
    def forward(self, T_vis: torch.Tensor, t_cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: update t_cam
        q = self.norm_q(t_cam)              # (B, 1, D_s)
        kv = self.norm_kv(T_vis)            # (B, M, D_s)
        # MultiheadAttention with batch_first expects (B, S, D)
        t_cam_updated, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        t_cam = t_cam + self.drop(t_cam_updated)
        t_cam = t_cam + self.cross_mlp(self.norm_q(t_cam))
 
        # Self-attention: joint over [t_cam, T_vis]
        joint = torch.cat([t_cam, T_vis], dim=1)  # (B, 1 + M, D_s)
        joint_norm = self.norm_joint(joint)
        joint_updated, _ = self.self_attn(query=joint_norm, key=joint_norm, value=joint_norm, need_weights=False)
        joint = joint + self.drop(joint_updated)
        joint = joint + self.self_mlp(self.norm_joint(joint))
 
        # Split back
        t_cam = joint[:, :1, :]
        T_vis = joint[:, 1:, :]
        return T_vis, t_cam
 
 
class SpatialEncoder(nn.Module):
    """
    E_spl(·): N stacked CrossSelfBlocks producing (T_spl, \hat{t}_cam).
 
    Inputs:
    - T_vis: (B, M, D_s)
    - t_cam: optional (B, 1, D_s). If None, uses learnable camera token (broadcast to batch).
 
    Outputs:
    - T_spl: (B, M, D_s)
    - t_cam_hat: (B, 1, D_s)
    """
    def __init__(
        self,
        dim: int,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        p_choose: float = 0.5,
        init_cam_token: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
 
        if init_cam_token is None:
            # Learnable camera token t_cam ∈ R^{D_s}
            self.t_cam = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.t_cam, std=0.02)
        else:
            assert init_cam_token.shape == (1, 1, dim)
            self.t_cam = nn.Parameter(init_cam_token)
 
        self.blocks = nn.ModuleList([
            CrossSelfBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])
        self.b_dist = Bernoulli(p_choose)
 
    def forward(self, T_vis: torch.Tensor, T_dpt: Optional[torch.Tensor] = None, t_cam: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, D = T_vis.shape
        assert D == self.dim, f"Token dim mismatch: got {D}, expected {self.dim}"

        T_spl = T_vis
        if T_dpt is not None:
            T_spl += self.b_dist.sample(T_vis.shape).to(T_dpt.device) * T_dpt
        t_cam_hat = self.t_cam.expand(B, -1, -1) 
        if t_cam is not None:
            dis = self.b_dist.sample(t_cam.shape).to(t_cam.device)
            t_cam_hat = t_cam_hat * (1 - dis) + t_cam * dis

        for blk in self.blocks:
            T_spl, t_cam_hat = blk(T_spl, t_cam_hat)
 
        return T_spl, t_cam_hat
 



class DepthEncoder(nn.Module):
    """
    深度编码器 E_dpt(·)
    将 [D'_t || M^{dpt}] 通过 14x14 卷积 patchify 为 token 序列 T^{dpt} ∈ R^{M×Ds}。

    - kernel_size = stride = patch_size = 14
    - 如果 H/W 不是 14 的倍数，会自动 padding 到最近的上界，从而保证整除。

    Args:
        Ds: token embedding 维度
        patch_size: 分块尺寸，默认为 14
        in_ch: 输入通道数，默认 2 (深度+掩码)
        post_layers: patchify 后的 1x1 轻量卷积层数（提高表达但不改变空间分辨率）
    """
    def __init__(self, Ds: int, patch_size: int = 14, in_ch: int = 2, post_layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        # Patchify: kernel=stride=patch_size，使每个 patch 生成一个 Ds 维向量
        self.patch_embed = nn.Conv2d(in_ch, Ds, kernel_size=patch_size, stride=patch_size)
        blocks = []
        for _ in range(max(0, post_layers - 1)):
            blocks += [nn.Conv2d(Ds, Ds, kernel_size=3), nn.GELU()]
        self.norm = nn.LayerNorm(Ds)
        self.post = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.input_size = (224, 224)

    def forward(
        self,
        depth: torch.Tensor,  # [B, H, W]
        mask: torch.Tensor,        # [B, H, W]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Returns:
            Tdpt: [B, M, Ds]
            grid_hw: (Gh, Gw) -> token 网格大小，M = Gh * Gw
        """
        depth_norm, mask = self.normalize_depth(depth, mask, method="mean_valid")
        B, H, W = depth_norm.shape
        x = torch.stack([depth_norm, mask.to(depth_norm.dtype)], dim=1)  # [B, 2, H, W]
        x = self.patch_embed(x)  # [B, Ds, Gh, Gw], Gh=H//14(向上取整), Gw=W//14(向上取整)
        x = self.post(x)
        x = x.flatten(2).transpose(1, 2)  # [B, M, Ds]
        x = self.norm(x)
        return x

    def normalize_depth(
        self,
        depth: torch.Tensor,                 # [B, H, W]
        mask: Optional[torch.Tensor] = None, # [B, H, W], 1=valid, 0=invalid
        method: str = "mean_valid",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        B, H, W = depth.shape
        if mask is None:
            mask = torch.ones_like(depth)
        target_h, target_w = self.input_size

        if depth.shape[-2:] != (target_h, target_w):
            depth = F.interpolate(depth.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)
            mask = F.interpolate(mask.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)

        mask = (mask > 0).float()
        if method == "mean_valid":
            denom = (depth * mask).sum(dim=(-2, -1), keepdim=True) / (mask.sum(dim=(-2, -1), keepdim=True) + eps)
        elif method == "mean_all":
            denom = depth.mean(dim=(-2, -1), keepdim=True)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        return depth / (denom + eps), mask

class VGGTSpatialEncoding(nn.Module):
    """
    Full pipeline:
    - DINO tokenization: I_t → T_vis
    - Concatenate learnable camera token t_cam and encode via SpatialEncoder:
      (T_spl, \hat{t}_cam) = E_spl(T_vis, t_cam)
 
    Usage:
    encoder = VGGTSpatialEncoding()
    T_spl, t_cam_hat = encoder(images)  # images: (B, 3, H, W) in [0,1]
    """
    def __init__(
        self,
        dino_model_name: str = "vit_base_patch16_224.dino",
        encoder_depth: int = 6,
        num_heads: int = 12,            # ViT-B embed_dim=768 typically uses 12 heads
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained: bool = True,
        input_dim = None
    ):
        super().__init__()
        self.tokenizer = DINOVisualTokenizer(dino_model_name, pretrained=pretrained)
        D_s = self.tokenizer.embed_dim
        self.image_encoder = SpatialEncoder(
            dim=D_s,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.pool = nn.MaxPool1d(2, stride=3)
        self.camera_encoder = nn.Sequential(
            nn.Linear(7, D_s*2),
            nn.GELU(),
            nn.Linear(D_s*2, D_s*2),
            nn.GELU(),
            nn.Linear(D_s*2, D_s),
        )
        self.depth_encoder = DepthEncoder(Ds=D_s, patch_size=14, in_ch=2)
        if input_dim is not None:
            self.output_nn = nn.Linear(D_s, input_dim)
        else:
            self.output_nn = None
        self.encoder = SpatialEncoder(
            dim=D_s,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
 
    def forward(self, images: torch.Tensor, depth=None, depth_mask=None, camera=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: (B, 3, H, W), float in [0,1]
        returns:
          T_spl: (B, M, D_s)
          t_cam_hat: (B, 1, D_s)
        """
        
        T_vis = self.tokenizer(images)                # (B, M, D_s)

        if depth is not None and depth_mask is not None:
            T_dpt = self.depth_encoder(depth, depth_mask)
            print(T_vis.shape, T_dpt.shape)
        else:
            T_dpt = None

        if camera is not None:
            T_cam = self.camera_encoder(camera)
        else:
            T_cam = None

        T_spl, t_cam_hat = self.encoder(T_vis, T_dpt, T_cam)        # (B, M, D_s), (B, 1, D_s)
        T_spl = self.pool(T_spl.transpose(-1,-2)).transpose(-1,-2)
        if self.output_nn is not None:
            T_spl = self.output_nn(T_spl)
        return T_spl, t_cam_hat
        
 
 
# Example usage
if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
 
    # Load an example image and convert to tensor in [0,1]
    img_path = "/cephfs/shared/lmz/vggt/pics/imgs4/frame_00042.png"  # replace with your image path
    img = Image.open(img_path).convert("RGB")
    to_tensor = T.Compose([
        T.ToTensor(),  # [0,1]
    ])
    x = to_tensor(img).unsqueeze(0)  # (1, 3, H, W)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    model = VGGTSpatialEncoding(
        dino_model_name="vit_base_patch16_224.dino",
        encoder_depth=6,
        num_heads=12,
        pretrained=True,
    ).to(device, dtype=dtype)
 
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            T_spl, t_cam_hat = model(
                x.to(device), 
                camera=torch.randn(1,7).to(device),
                depth=x.mean(dim=1).to(device),
                depth_mask=x.mean(dim=1).to(device)/2
            )
            print(f"T_spl shape: {T_spl.shape}")       # (B, M, D_s)
            print(f"t_cam_hat shape: {t_cam_hat.shape}")  # (B, 1, D_s)