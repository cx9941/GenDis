import torch
import torch.nn.functional as F
import torch.nn as nn

class CCAProjection(nn.Module):
    """
    把两路 hidden state 投影到同一 k 维子空间，返回
      z1, z2 : 对齐后的两路表征
      com    : 融合得到的公共表征 (这里采用简单平均，也可 concat/加权)
    """
    def __init__(self, in_dim1, in_dim2, k):
        super().__init__()
        self.proj1 = nn.Linear(in_dim1, k, bias=False)
        self.proj2 = nn.Linear(in_dim2, k, bias=False)
        # self.layernorm1 = nn.LayerNorm(k)
        # self.layernorm2 = nn.LayerNorm(k)
        self.outdim_size=k

    def forward(self, h1, h2):
        z1 = self.proj1(h1)
        z2 = self.proj2(h2)
        # z1 = self.layernorm1(z1)
        # z2 = self.layernorm2(z2)
        com = torch.concat([z1,z2], dim=-1)
        return z1, z2, com
    
    def calculate_loss(self, H1, H2, r1=1e-3, r2=1e-3, eps=1e-9, mode=0):
        # print(H1.size())
        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        # SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        # SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,H2bar.t()) + r2 * torch.eye(o2, device=H1.device)
        # SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,H1bar.t()) + r1 * torch.eye(o1, device=H1.device)

        SigmaHat11 = H1bar @ H1bar.T / (m - 1) + r1 * torch.eye(o1, device=H1.device)
        SigmaHat22 = H2bar @ H2bar.T / (m - 1) + r2 * torch.eye(o2, device=H1.device)
        SigmaHat12 = H1bar @ H2bar.T / (m - 1)

        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition

        # print("SigmaHat11")
        # print(SigmaHat11)

        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        
        D1 = torch.clamp(D1, min=1e-6).to(H1.device)
        D2 = torch.clamp(D2, min=1e-6).to(H1.device)

        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1].to(H1.device)
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2].to(H1.device)
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        # print(SigmaHat11RootInv.dtype)
        # print(SigmaHat12.dtype)
        # print(SigmaHat22RootInv.dtype)
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12.to(torch.float32)), SigmaHat22RootInv)

        # just the top self.outdim_size singular values are used

        # print(Tval.shape)
        # print(Tval)

        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(H1.device)) # regularization for more stability
        
        # print(trace_TT)
        
        U, V = torch.linalg.eigh(trace_TT)
        U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(H1.device))
        
        U = torch.clamp(U, min=1e-6)

        U = U.topk(self.outdim_size)[0]

        if mode == 2:
            corr = torch.mean(torch.log(U))
            return -corr
        elif mode == 1:
            corr = torch.mean(torch.sqrt(U))
            return -corr
        elif mode == 0:
            corr = torch.sum(torch.sqrt(U))
            return -corr
        else:
            assert False



    def Soft_DCCA_loss(self, view_t1: torch.Tensor, view_t2: torch.Tensor, eta1=1e-2, eta2=1e-3, rho=0.9, 
                    S1_prev=None, S2_prev=None):
        """
        PyTorch version of Soft DCCA loss (CVPR18).
        
        Args:
            view_t1: Tensor of shape (n_samples, n_bands)
            view_t2: Tensor of shape (n_samples, n_bands)
            eta1: weight for correlation loss
            eta2: weight for decorrelation loss
            rho: decay factor for covariance matrix smoothing
            S1_prev: previous covariance matrix for view_t1 (torch.Tensor or None)
            S2_prev: previous covariance matrix for view_t2 (torch.Tensor or None)

        Returns:
            corr_loss: torch scalar tensor
            decov_loss: torch scalar tensor
            S1_updated, S2_updated: updated covariance matrices (to be reused if needed)
        """
        # 1. Correlation loss (L2 distance between the two views)
        corr_square = (view_t1 - view_t2) ** 2
        corr_loss = torch.sqrt(torch.sum(corr_square, dim=-1) + 1e-8)  # Avoid sqrt(0)
        corr_loss = eta1 * torch.mean(corr_loss)

        # 2. Decorrelation loss
        N1, d1 = view_t1.shape
        N2, d2 = view_t2.shape

        S_t1 = (view_t1.T @ view_t1) / (N1 - 1)
        S_t2 = (view_t2.T @ view_t2) / (N2 - 1)

        # Exponential moving average of covariance matrices
        if S1_prev is None:
            S1_prev = torch.zeros_like(S_t1, device=view_t1.device)
        if S2_prev is None:
            S2_prev = torch.zeros_like(S_t2, device=view_t2.device)

        S1_updated = rho * S1_prev + (1 - rho) * S_t1
        S2_updated = rho * S2_prev + (1 - rho) * S_t2

        decov_loss_t1 = torch.sum(torch.abs(S1_updated)) - torch.sum(torch.diag(S1_updated))
        decov_loss_t2 = torch.sum(torch.abs(S2_updated)) - torch.sum(torch.diag(S2_updated))
        decov_loss = eta2 * torch.mean(decov_loss_t1 + decov_loss_t2)

        return corr_loss, decov_loss, S1_updated.detach(), S2_updated.detach()