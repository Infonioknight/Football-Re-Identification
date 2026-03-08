import numpy as np
import torch
import torchreid
from torchvision import transforms
from PIL import Image

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = torchreid.models.build_model(
    name="osnet_ain_x1_0",
    num_classes=1000,  
    pretrained=True,    
)
_model.eval().to(_device)

_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(512, dtype=np.float32)

    img    = Image.fromarray(crop_bgr[:, :, ::-1])   
    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        feat = _model(tensor).squeeze(0).cpu().numpy()

    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat /= norm
    return feat.astype(np.float32)
