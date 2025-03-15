import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, D, N, N]
        x = x.flatten(2).transpose(1, 2)  # [B, D, N, N] -> [B, N, D]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=256, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(emb_dim * mlp_ratio), emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class LightweightViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_dim=256, num_layers=4, num_heads=4, out_dim=384):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_dim))
        self.encoder = nn.Sequential(*[TransformerEncoder(emb_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, out_dim)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # 取 CLS token
        return self.head(x)




class SimpleCNN(nn.Module):
    def __init__(self, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Linear(512 * 14 * 14, output_dim)  # 经过 4 层池化后，图片尺寸变为 14x14

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 56x56
        x = self.pool(F.relu(self.conv3(x)))  # 28x28
        x = self.pool(F.relu(self.conv4(x)))  # 14x14

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    


# 测试模型
if __name__ == "__main__":
    model = LightweightViT()
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print("Output shape:", output.shape)  # 应该是 [1, 16]
    model = SimpleCNN(output_dim=128)
    sample_input = torch.randn(1, 3, 224, 224)  # 模拟一张图片
    output = model(sample_input)
    print(output.shape)  # 预期输出: torch.Size([1, 128])
