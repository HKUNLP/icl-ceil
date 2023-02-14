import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor as T
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class BiEncoderConfig(PretrainedConfig):
    model_type = "BiEncoder"

    def __init__(
            self,
            q_model_name=None,
            ctx_model_name=None,
            ctx_no_grad=True,
            margin=0.2,
            scale_factor=0.1,
            pair_wise=False,
            dpp_training=False,
            norm_embed=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_model_name = q_model_name
        self.ctx_model_name = ctx_model_name
        self.ctx_no_grad = ctx_no_grad
        self.margin = margin
        self.scale_factor = scale_factor
        self.pair_wise = pair_wise
        self.norm_embed = norm_embed
        self.dpp_training = dpp_training


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BiEncoder(PreTrainedModel):
    config_class = BiEncoderConfig

    def __init__(self, config):
        super(BiEncoder, self).__init__(config)
        assert config.q_model_name is not None or config.ctx_model_name is not None

        if config.q_model_name is not None:
            self.question_model = AutoModel.from_pretrained(config.q_model_name)
        else:
            self.question_model = None

        if config.ctx_model_name is not None:
            self.ctx_model = AutoModel.from_pretrained(config.ctx_model_name)
        else:
            self.ctx_model = None

        # share q and ctx model if one of them is None
        if self.question_model is None and self.ctx_model is not None:
            self.question_model = self.ctx_model
            logging.info("Sharing ctx_model with question_model")
        if self.question_model is not None and self.ctx_model is None:
            self.ctx_model = self.question_model
            logging.info("Sharing question_model with ctx_model")

        self.ctx_no_grad = config.ctx_no_grad
        self.norm_embed = config.norm_embed
        self.scale_factor = config.scale_factor
        if config.dpp_training:
            self.loss_func = self.calc_dpp_loss
        else:
            self.loss_func = self.calc_nll_loss

        self.pair_wise = config.pair_wise
        self.margin = config.margin

    def encode(self, input_ids, attention_mask, encode_ctx=False, **kwargs):
        if encode_ctx:
            if self.ctx_no_grad:
                with torch.no_grad():
                    enc_emb = self.ctx_model(input_ids, attention_mask)
            else:
                enc_emb = self.ctx_model(input_ids, attention_mask)
        else:
            enc_emb = self.question_model(input_ids, attention_mask)
        enc_emb = mean_pooling(enc_emb, attention_mask)
        if self.norm_embed:
            enc_emb = enc_emb / enc_emb.norm(p=2, dim=-1, keepdim=True)
        return enc_emb

    def forward(
            self,
            questions_tensor: T,
            questions_attn_mask: T,
            ctxs_tensor: T,
            ctxs_attn_mask: T,
            ctx_indices: T,
            labels
    ) -> Dict:
        q_pooled_out = self.encode(questions_tensor, questions_attn_mask, encode_ctx=False)
        ctx_pooled_out = self.encode(ctxs_tensor, ctxs_attn_mask, encode_ctx=True)
        return self.loss_func(q_pooled_out, ctx_pooled_out, ctx_indices, labels)

    def calc_nll_loss(
            self,
            q_vectors: T,
            ctx_vectors: T,
            ctx_indices: T,
            labels: Optional[T],
    ) -> Dict:
        assert ctx_indices.shape[1] == 1, "In-context number != 1, set dpp_training to true!"
        scores = torch.matmul(q_vectors, ctx_vectors.T)

        if not self.pair_wise:
            # directly get pos_idx in ctx_vectors
            labels = ctx_indices.squeeze(1)[labels]
            softmax_scores = F.log_softmax(scores, dim=1)

            loss = F.nll_loss(
                softmax_scores,
                labels,
                reduction="mean",
            )
        else:
            batch_size = scores.shape[0]
            # batch, num_hard_pos_neg
            ctx_indices = ctx_indices.reshape(-1).reshape(batch_size, -1)
            hard_pos_neg_num = ctx_indices.shape[1]
            in_batch_neg_num = hard_pos_neg_num
            full_ctx_indices = []
            for i in range(batch_size):
                neg_ctx_indices = torch.cat([ctx_indices[:i], ctx_indices[i + 1:]], dim=0).reshape(-1)
                rand_indx = torch.randperm(len(neg_ctx_indices))
                neg_ctx_indices = neg_ctx_indices[rand_indx][:in_batch_neg_num]
                per_sample_ctx_indices = torch.cat([ctx_indices[i], neg_ctx_indices], dim=0)
                full_ctx_indices.append(per_sample_ctx_indices)

            full_ctx_indices = torch.stack(full_ctx_indices, dim=0)
            scores = scores.gather(-1, full_ctx_indices)

            loss = ranking_loss(scores[:, :hard_pos_neg_num], margin=self.margin)
        return {'loss': loss, 'logits': scores}

    def calc_dpp_loss(
            self,
            q_vectors: T,
            ctx_vectors: T,
            ctx_indices: T,  # batch*(1+neg), num_ice
            labels: Optional[T]
    ) -> Dict:
        """
        Computes dpp loss for the given of question and ctx vectors.
        :return: a dict of loss value and logits
        """
        batch_size = q_vectors.shape[0]
        num_all_ctx = ctx_vectors.shape[0]
        batch_size_mul_num_pos_neg, num_ice = ctx_indices.shape
        num_pos_neg = batch_size_mul_num_pos_neg // batch_size

        # batch, num_all_ctx
        rel_scores = torch.matmul(q_vectors, ctx_vectors.T)
        # to make kernel-matrix non-negative
        rel_scores = (rel_scores + 1) / 2
        # to prevent overflow error
        rel_scores = rel_scores - rel_scores.max(dim=-1, keepdim=True)[0].detach()
        # to balance relevance and diversity
        rel_scores = (rel_scores / (2 * self.scale_factor)).exp()
        # num_all_ctx, num_all_ctx
        kernel_matrix = torch.matmul(ctx_vectors, ctx_vectors.T)
        # to make kernel-matrix non-negative
        kernel_matrix = (kernel_matrix + 1) / 2
        # batch, num_all_ctx, num_all_ctx
        kernel_matrix = rel_scores[:, None] * kernel_matrix[None] * rel_scores[..., None]
        # batch, num_pos_neg, num_ice
        ctx_indices = ctx_indices.reshape(batch_size, num_pos_neg, num_ice)

        in_batch_neg_num = num_pos_neg
        scores = []
        for i in range(batch_size):
            per_sample_kernel_matrix = kernel_matrix[i]
            # num_pos_neg, num_ice   ignore in-batch negative
            per_sample_ctx_indices = ctx_indices[i].reshape(-1, num_ice)
            # num_pos_neg, num_ice, num_ice
            per_sample_neg_submatrix = indexing(per_sample_kernel_matrix, per_sample_ctx_indices)
            # num_pos_neg
            per_sample_scores = torch.linalg.slogdet(per_sample_neg_submatrix).logabsdet
            scores.append(per_sample_scores)

        scores = torch.stack(scores, dim=0)
        if not self.pair_wise:
            scores = scores[:, :num_pos_neg]
            shifted_scores = scores - scores.max(dim=-1, keepdim=True)[0]
            softmax_scores = F.log_softmax(shifted_scores, dim=1)
            # info-nce loss
            loss = F.nll_loss(
                softmax_scores,
                labels,
                reduction="mean",
            )
        else:
            loss = ranking_loss(scores[:, :num_pos_neg], margin=self.margin)

        if loss.isnan() or loss.isinf() or loss == 0:
            print("inf in kernel_matrix?", kernel_matrix.isinf().any())
            print("loss", loss)
            print("scores", scores)
            print("min score", scores.min())
            print("inf in score?", scores.isinf().any())
            print("nan in score?", scores.isnan().any())
            # exit()
        return {'loss': loss, 'logits': scores}


def create_indices(indices):
    '''

    Args:
        indices: tensor with size [batch, n]

    Returns:
        tensor with size [batch, n, n, 2]

        Examples:
            input: tensor([[1,2,3]])
            output:
                tensor([[[[1, 1],
                          [1, 2],
                          [1, 3]],
                         [[2, 1],
                          [2, 2],
                          [2, 3]],
                         [[3, 1],
                          [3, 2],
                          [3, 3]]]])
    '''
    n = indices.shape[1]
    indices = indices.unsqueeze(-1)
    indices1 = indices.unsqueeze(2).repeat(1, 1, n, 1)
    indices2 = indices.unsqueeze(1).repeat(1, n, 1, 1)
    res = torch.cat([indices1, indices2], -1)
    return res


def indexing(S, indices):
    '''

    Args:
        S: tensor with size [b, N, N]
        indices: tensor with size [b, n, n, 2] or [b, n]

    Returns:
        tensor with size [b, n, n]
    '''
    batch_size, n = indices.shape
    if len(indices.shape) == 2:
        # enumerate all combinations
        indices = create_indices(indices)

    all_i = indices[..., 0].reshape(-1)
    all_j = indices[..., 1].reshape(-1)
    sub_S = S[all_i, all_j]
    return sub_S.reshape(batch_size, n, n)


def ranking_loss(cand_scores, margin=0.):
    batch, n = cand_scores.size()
    total_loss = 0
    # normalize to stabilize the margin-based loss
    max_score = cand_scores.max(dim=-1, keepdim=True)[0].detach()
    min_score = cand_scores.min(dim=-1, keepdim=True)[0].detach()
    cand_scores = (cand_scores - min_score) / (max_score-min_score)
    for i in range(1, n):
        pos_score = cand_scores[:, :-i]
        neg_score = cand_scores[:, i:]
        loss_func = torch.nn.MarginRankingLoss(margin * i)  # batch x i
        ones = torch.ones(pos_score.size(), device=pos_score.device)
        total_loss += loss_func(pos_score, neg_score, ones)
    return total_loss
