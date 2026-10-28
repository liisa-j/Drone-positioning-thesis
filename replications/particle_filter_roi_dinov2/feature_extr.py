import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

# runs model, processes output, reshapes into a grid
class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        """
        x => (B,3,H,W)
        esmalt forward output_hidden_states=True,
        siis reshape last_hidden_state => (B, C, h, w).
        """
        outputs = self.backbone(x, output_hidden_states=True)
        hidden = outputs.last_hidden_state   #> (B, tokens, hidden_dim)

        B, tokens, dim = hidden.shape

        # Drop CLS token (always at index 0)
        hidden_no_cls = hidden[:, 1:, :]

        # Compute expected patch tokens from input image size
        patch_size = self.backbone.config.patch_size
        h_patches = x.shape[2] // patch_size
        w_patches = x.shape[3] // patch_size
        expected_tokens = h_patches * w_patches

        # Debug prints
        """print(f"[DEBUG] Input image: {x.shape[2]}x{x.shape[3]}")
        print(f"[DEBUG] Patch size: {patch_size}")
        print(f"[DEBUG] Expected tokens (H*W): {expected_tokens}")
        print(f"[DEBUG] Actual tokens from model: {tokens}")"""

        # If we have more tokens than patches, drop the extra ones (registers)
        if hidden_no_cls.shape[1] > expected_tokens:
            #print(f"[DEBUG] Dropping {tokens - expected_tokens} register tokens")
            hidden_no_cls = hidden_no_cls[:, :expected_tokens, :]

        # Now reshape into feature map
        feats = hidden_no_cls.reshape(B, h_patches, w_patches, dim)
        feats = feats.permute(0, 3, 1, 2)  # (B, C, H, W)

        #print(f"[DEBUG] Final feature map shape: {feats.shape}")
        return feats


#transforms
map_transform= T.Compose([
    T.Resize((1024,1024)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

drone_transform= T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# loads model, prepares image
class DinoExtractor:
    def __init__(self, model_name):
        """
        Load a DINO model (v2 or v3) and prepare for feature extraction.
        Each instance keeps its own model and device.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Loading model: facebook/{model_name}")

        # Load model
        self.model = AutoModel.from_pretrained(
            f"facebook/{model_name}",
            trust_remote_code=True,
        ).eval().to(self.device)


        self.extractor = DinoV3FeatureExtractor(self.model).to(self.device)

        print("Model and feature extractor ready.")

    @torch.no_grad()
    def extract_features(self, pil_img: Image.Image, transform):
        """
        Extract features for a given image.
        """
        x = transform(pil_img).unsqueeze(0).to(self.device)
        feats = self.extractor(x)
        return feats

    def unload(self):
        """
        Free GPU memory.
        """
        del self.model, self.extractor
        torch.cuda.empty_cache()