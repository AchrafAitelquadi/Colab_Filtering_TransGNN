import torch as t
import torch.nn.functional as F


# ============================
# 1) Produit scalaire user–item
# ============================
def innerProduct(user_emb, item_emb):
    """
    Compute dot-product score between user and item embeddings.
    Equivalent to P(u, i) = <h_u, h_i>.
    
    user_emb : (B, d)
    item_emb : (B, d)
    
    return    : (B,) similarity score
    """
    return t.sum(user_emb * item_emb, dim=-1)



# =====================================
# 2) Pairwise score for BPR / ranking loss
# =====================================
def pairPredict(usr_emb, pos_emb, neg_emb):
    """
    Score difference : P(u, i_pos) - P(u, i_neg)
    Used by pairwise ranking loss (BPR style).
    
    usr_emb  : (B, d)
    pos_emb  : (B, d)
    neg_emb  : (B, d)
    """
    pos_score = innerProduct(usr_emb, pos_emb)
    neg_score = innerProduct(usr_emb, neg_emb)
    return pos_score - neg_score



# ============================
# 3) L2 Regularization
# ============================
def calcRegLoss(model):
    """
    Compute L2 norm over all trainable parameters (weight decay).
    """
    reg = 0.0
    for W in model.parameters():
        reg += W.norm(2).pow(2)
    return reg



# =============================================
# 4) Contrastive Loss (SimCLR / InfoNCE style)
# =============================================
def contrastLoss(emb1, emb2, nodes, temp=0.2):
    """
    Contrastive loss between two embedding views (emb1, emb2).
    
    emb1, emb2 : (N, d) embedding matrices
    nodes      : indices of nodes to use in the batch
    temp       : temperature scalar

    Implements: 
        -log( exp(sim(u1,u2)/T) / sum(exp(sim(u1, all_emb2))/T) )
    """

    # Normalize for cosine similarity (stabilizes training)
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)

    # Pick only embeddings of sampled nodes
    z1 = emb1[nodes]    # (B, d)
    z2 = emb2[nodes]    # (B, d)

    # --- Positive similarity -----------------------------
    # sim(z1, z2) for the SAME node
    pos_sim = t.sum(z1 * z2, dim=-1) / temp
    numerator = t.exp(pos_sim)

    # --- Negative / all similarities ----------------------
    # Compute similarity with EVERY embedding of view 2
    # z1 @ emb2.T -> shape (B, N)
    all_sim = (z1 @ emb2.T) / temp
    denominator = t.exp(all_sim).sum(dim=-1) + 1e-8

    # InfoNCE loss
    loss = -t.log(numerator / denominator).mean()
    return loss
