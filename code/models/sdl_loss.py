import torch
import torch.nn as nn

class SDLLoss(nn.Module):
    def __init__(self, feature_dim, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("C_accu", torch.zeros(feature_dim, feature_dim))
        self.register_buffer("c", torch.tensor(0.0))

    def forward(self, Z):
        m = Z.size(0)
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        C_mini = (Z_centered.T @ Z_centered) / (m - 1)

        self.C_accu = self.alpha * self.C_accu + C_mini.detach()
        self.c = self.alpha * self.c + 1.0
        C_appx = self.C_accu / self.c

        off_diag = C_appx - torch.diag(torch.diag(C_appx))
        loss = torch.sum(torch.abs(off_diag))
        return loss

class SoftCCALoss(nn.Module):
    def __init__(self, feature_dim, lambda_sdl=1.0):
        super().__init__()
        self.lambda_sdl = lambda_sdl
        self.sdl1 = SDLLoss(feature_dim)
        self.sdl2 = SDLLoss(feature_dim)

    def forward(self, z1, z2):
        # align_loss = torch.sqrt(((z1 - z2) ** 2)).mean()
        # align_loss = torch.norm(z1 - z2, p=2).mean()
        import torch.nn.functional as F

        # 用 L2 范数进行标准化（每个样本单位范数）
        norm_z1 = F.normalize(z1, dim=1)
        norm_z2 = F.normalize(z2, dim=1)
        
        corr_square = (norm_z1 - norm_z2) ** 2
        align_loss = torch.sqrt(torch.sum(corr_square, dim=-1) + 1e-8).mean()  # Avoid sqrt(0)
        sdl_loss1 = self.sdl1(norm_z1)
        sdl_loss2 = self.sdl2(norm_z2)
        total_loss = align_loss + self.lambda_sdl * (sdl_loss1 + sdl_loss2)
        return total_loss