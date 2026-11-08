import torch
import torch.nn.functional as F
import math


class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_residual: bool = False):
        """
        Q in R^{..., N_q x d}
        K, V in R^{..., N_k x d}
        """
        factor = math.sqrt(Q.size(dim=-1))
        *input_shape, seq_len_q, d = Q.size()

        q_tile_size = 32
        k_tile_size = 32

        T_q = math.ceil(Q.size(dim=-2) / q_tile_size)
        T_k = math.ceil(K.size(dim=-2) / k_tile_size)

        q_tiles = torch.split(F.pad(Q, (0, 0, 0, q_tile_size * T_q - Q.size(dim=-2))), q_tile_size, dim=-2)
        k_tiles = torch.split(F.pad(K, (0, 0, 0, k_tile_size * T_k - K.size(dim=-2))), k_tile_size, dim=-2)
        v_tiles = torch.split(F.pad(V, (0, 0, 0, k_tile_size * T_k - K.size(dim=-2))), k_tile_size, dim=-2)

        outputs = []
        Ls = []

        for i in range(T_q):
            Q_i = q_tiles[i]
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros((*input_shape, q_tile_size), dtype=Q.dtype, device=Q.device)
            m_i = torch.ones((*input_shape, q_tile_size), dtype=Q.dtype, device=Q.device) * (-torch.inf)

            for j in range(T_k):
                K_j, V_j = k_tiles[j], v_tiles[j]
                S_ij = Q_i @ K_j.transpose(-1, -2) / factor

                if j == T_k - 1 and k_tile_size * T_k != K.size(dim=-2):
                    # print("YEY")
                    S_ij[..., -(k_tile_size * T_k - K.size(dim=-2)):] = -torch.inf

                m_i_candidate = S_ij.max(dim=-1).values
                m_i_prev = m_i
                m_i = torch.maximum(m_i, m_i_candidate)
                P_ij = torch.exp(S_ij - m_i.unsqueeze(dim=-1))

                exp_max_diff = torch.exp(m_i_prev - m_i)  # [..., q_tile_size]
                l_i = l_i * exp_max_diff + torch.sum(P_ij, dim=-1)
                O_i = exp_max_diff.unsqueeze(dim=-1) * O_i + P_ij @ V_j

            O_i = (1 / l_i).unsqueeze(dim=-1) * O_i
            L_i = m_i + torch.log(l_i)

            outputs.append(O_i)
            Ls.append(L_i)

        O = torch.cat(outputs, dim=-2)[..., :seq_len_q, :]
        L = torch.cat(Ls, dim=-1)[..., :seq_len_q, :]

        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        return super().backward(ctx, *grad_outputs)

# x = torch.ones((12, 16))
# *input_shape, seq_len, d = x.size()
# print(input_shape)
# print(seq_len)
# print(d)
# B, T = 4, 3
# x_padded = F.pad(x, (0, 0, 0, B * T - x.size(0)))  #  value=-torch.inf
# print(x_padded.shape)
# print(x_padded[-3:, :])
# x_padded[-(B * T - 12):] = 666
# print(x_padded)

# x_rand = torch.randint(0, 100, (3, 5))
# print(x_rand)
# print(x_rand.max(dim=1).values)