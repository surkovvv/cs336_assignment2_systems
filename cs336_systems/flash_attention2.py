import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl


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

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) + (-float('inf'))
    m_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) + (-float('inf'))
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # intermediate_result = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        intermediate_result = tl.dot(q_tile, tl.trans(k_tile)) * scale
        m_prev = m
        m = tl.maximum(m, tl.max(intermediate_result, axis=1))

        intermediate_result = tl.exp(intermediate_result - m[None])
        l = tl.exp(m_prev - m) * l + tl.sum(intermediate_result, axis=1)
        if j == tl.cdiv(N_KEYS, K_TILE_SIZE) - 1:
            intermediate_result = ... # TODO - masking bro!!
        
        output = tl.exp(m_prev - m)[None] * output + tl.dot(intermediate_result.to(v_tile.dtype), v_tile)

        # advance v and k tiles
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    output = output / l[None]
    l = m + tl.log(l)
    
    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))

        