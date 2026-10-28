import os
from huggingface_hub import hf_hub_download

IMAGENET_REPO = "nvidia/PixelDiT-ImageNet"

KNOWN = {
    "imagenet256_pixeldit_xl_epoch80.ckpt": IMAGENET_REPO,
    "imagenet256_pixeldit_xl_epoch160.ckpt": IMAGENET_REPO,
    "imagenet256_pixeldit_xl_epoch320.ckpt": IMAGENET_REPO,
    "imagenet512_pixeldit_xl.ckpt": IMAGENET_REPO,
}

WID_REPO = "ziqiaow/Wiener-informed-Decoding"
KNOWN.update({f"imagenet256_wid_pixeldit_xl_epoch{epoch}.ckpt": WID_REPO
              for epoch in (80, 160, 260, 320, 500)})
KNOWN["imagenet512_wid_pixeldit_xl_epoch50.ckpt"] = WID_REPO


def resolve_checkpoint(path):
    if not path or os.path.exists(path):
        return path
    name = os.path.basename(path)
    if name in KNOWN:
        return hf_hub_download(KNOWN[name], name)
    return path
