
import open_clip
import torch
import torch.nn as nn
from transformers import Blip2Model, CLIPModel, ViTModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

model_name = "openai/clip-vit-base-patch16"


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size):
        """
        A single transformer block.

        Args:
            hidden_size (int): Input and output dimension of the tokens.
            num_heads (int): Number of attention heads.
            ff_hidden_size (int): Dimension of the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        # Layer normalization for attention
        self.norm1 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.GELU(),
            nn.Linear(ff_hidden_size, hidden_size)
        )

        # Layer normalization for FFN
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, hidden_size).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)  # Residual connection

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual connection

        # print(x.shape)

        return x


class ViTWithTransformerBlocks(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224",
                 hidden_size=768,
                 ff_hidden_size=3072,
                 num_heads=8,
                 embedding_dim=768):
        """
        Vision Transformer with two additional transformer blocks for feature refinement.

        Args:
            pretrained_model_name (str): Name of the pretrained ViT model.
            hidden_size (int): Input dimension for the transformer blocks.
            ff_hidden_size (int): Hidden size for the feed-forward network.
            num_heads (int): Number of attention heads in the transformer blocks.
            embedding_dim (int): Output embedding dimension.
        """
        super(ViTWithTransformerBlocks, self).__init__()

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Two custom transformer blocks
        self.transformer_block1 = TransformerBlock(
            hidden_size=hidden_size, num_heads=num_heads, ff_hidden_size=ff_hidden_size)
        self.transformer_block2 = TransformerBlock(
            hidden_size=hidden_size, num_heads=num_heads, ff_hidden_size=ff_hidden_size)

        # Fully connected layer for projection
        self.fc = nn.Linear(hidden_size*2, embedding_dim)
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            embeddings (torch.Tensor): Normalized feature embeddings.
            norms (torch.Tensor): L2 norms of the embeddings.
        """
        # Extract class token and patch embeddings
        outputs = self.vit(x)  # (batch_size, num_tokens, hidden_size)

        # First transformer block
        # (batch_size, num_tokens, hidden_size)
        refined_tokens1 = self.transformer_block1(outputs.last_hidden_state)

        # Second transformer block
        # (batch_size, num_tokens, hidden_size)
        refined_tokens2 = self.transformer_block2(refined_tokens1)

        # Extract refined class token and refined patch embeddings
        # (batch_size, hidden_size)
        refined_cls_token = refined_tokens2[:, 0, :]
        # (batch_size, num_patches, hidden_size)
        refined_patch_embeddings = refined_tokens2[:, 1:, :]

        # Average pooling over patch embeddings
        avg_pooled_patches = torch.mean(
            refined_patch_embeddings, dim=1)  # (batch_size, hidden_size)
        # print(avg_pooled_patches.shape)

        # Concatenate refined class token and average pooled patches
        concatenated_features = torch.cat(
            # (batch_size, hidden_size * 2)
            [refined_cls_token, avg_pooled_patches], dim=1)
        # print(concatenated_features.shape)

        # Project to embedding space
        # (batch_size, embedding_dim)
        embeddings = self.fc(concatenated_features)

        # Normalize embeddings
        embeddings = self.l2_norm(embeddings, p=2, dim=1)

        # Compute norms
        norms = torch.norm(embeddings, 2, dim=1, keepdim=True)

        return embeddings, norms


class FRfromVLM(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        """
        Initialize the FRfromVLM model with a pre-trained CLIP model.

        Args:
            model_name (str): Name of the pre-trained CLIP model.
        """
        super(FRfromVLM, self).__init__()
        self.model_name = model_name
        if "OpenCLIP" in model_name:
            model_id = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            model, _, processor = open_clip.create_model_and_transforms(
                model_id)
            self.backbone = model

        elif "clip" in model_name:
            # Load the CLIP model
            self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
        elif "blip2" in model_name:
            # Load the BLIP-2 model
            from transformers import Blip2Model
            self.backbone = Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b').vision_model

    def forward(self, x):
        """
        Forward pass through the CLIP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            embeddings (torch.Tensor): Feature embeddings from the CLIP model.
            norms (torch.Tensor): L2 norms of the embeddings.
        """
        if "OpenCLIP" in self.model_name:
            # Preprocess the input for OpenCLIP
            x = self.backbone.encode_image(x)  # (batch_size, hidden_size)
        else:
            # (batch_size, num_tokens, hidden_size)
            outputs = self.backbone(pixel_values=x).last_hidden_state
            # Extract the class token (first token) and patch embeddings
            x = outputs[:, 0, :]  # (batch_size, hidden_size)

        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm


def load_model_by_name(model_name="FineTuned/OpenCLIP-Huge"):
    """
    Load a model based on the provided model name.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        nn.Module: The loaded model.
    """
    supported_models = ["FineTuned/OpenCLIP-Huge", "FineTuned/blip2-opt-2.7b", "FineTuned/clip-vit-base-patch16"]
    if model_name in supported_models:
        model = FRfromVLM(model_name=model_name).eval()
        model_weight_path = f"/mnt/scratch/sonymd/AFRvsVLM/weights/{model_name}/last.ckpt"
        state_dict = torch.load(model_weight_path, map_location="cpu")[
            'state_dict']

        # Remove 'model.' prefix from state_dict keys
        state_dict = {k.replace('model.', ''): v for k,
                      v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return model
    else:
        raise ValueError(f"Model {model_name} is not supported. Supported models: {supported_models}")


if __name__ == "__main__":

    # Test all models
    for model_name in vit_model_names_224:
        print(f"Testing model: {model_name}")

        # Get configuration for the model
        config = get_vit_specific_config_224(model_name)
        # Setting embedding dimension for all models
        config['embedding_dim'] = 512

        # Initialize the model
        model = ViTWithTransformerBlocks(
            pretrained_model_name=model_name,
            hidden_size=config["hidden_size"],
            ff_hidden_size=config["ff_hidden_size"],
            num_heads=config["num_heads"],
            embedding_dim=config["embedding_dim"]
        )

        # Set device
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Dummy input tensor
        # Batch size = 2, Image size = 224x224
        x = torch.randn(2, 3, 224, 224).to(device)

        # Forward pass
        embeddings, norms = model(x)

        # Print results
        print(f"Results for {model_name}:")
        print("Embeddings shape:", embeddings.shape)  # Expected: (2, 512)
        print("Norms shape:", norms.shape)            # Expected: (2, 1)
        print("=" * 50)
