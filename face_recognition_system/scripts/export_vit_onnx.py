from pathlib import Path
import argparse

import torch
import torch.nn as nn
import timm


class ViTAntiSpoofing(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.vit(x)
        return self.classifier(feat)


def load_state_dict(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ViT anti-spoof model to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to vit_best.pth")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model = ViTAntiSpoofing(num_classes=2, dropout_rate=0.3)
    load_state_dict(model, ckpt)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes=None,
    )
    print(f"Saved ONNX to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
