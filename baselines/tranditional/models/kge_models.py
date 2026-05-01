import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseKGE(nn.Module):
    def __init__(self, args):
        super(BaseKGE, self).__init__()
        self.args = args
        self.num_ent = args.num_ent
        self.num_rel = args.num_rel
        self.emb_dim = args.emb_dim
        
        self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim)
        self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def forward(self, sub, rel, obj=None):
        raise NotImplementedError

class TransE(BaseKGE):
    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.gamma = args.margin # Using margin as gamma for distance offset
        
    def forward(self, sub, rel, obj=None):
        # sub: (batch,) indices
        # rel: (batch,) indices
        
        # [MODIFIED] HoGRN TransE setting appears to NOT use L2 normalization
        # given the Gamma=40.0 parameter which would saturate normalized vectors.
        # We rely on optimizer/loss to regulate magnitudes.
        h = self.ent_emb(sub)
        r = self.rel_emb(rel)

        if obj is not None:
            # Training Mode (Pairwise Score)
            # obj: (batch,) indices
            t = self.ent_emb(obj)
            
            # Distance: ||h + r - t||
            # We naturally want to minimize this for positive, maximize for negative
            # Score = gamma - distance (Higher is better)
            score = self.gamma - torch.norm(h + r - t, p=1, dim=1)
            return score
        else:
            # Inference Mode / 1-N Training Mode
            # Compute score against ALL candidates
            # h: (B, D), r: (B, D)
            # query: (B, 1, D)
            query = (h + r).unsqueeze(1)
            
            # all_ent: (N, D)
            all_ent = self.ent_emb.weight # No Norm
            all_ent = all_ent.unsqueeze(0) # (1, N, D)
            
            # Distance L1: || (h+r) - t_all ||_1
            # (B, 1, D) - (1, N, D) -> (B, N, D) -> norm -> (B, N)
            dist = torch.norm(query - all_ent, p=1, dim=2)
            
            score = self.gamma - dist
            # HoGRN uses Sigmoid for 1-vs-All BCELoss
            return torch.sigmoid(score)

class DistMult(BaseKGE):
    def __init__(self, args):
        super(DistMult, self).__init__(args)
        # HoGRN uses a bias term for all entities in DistMult
        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def forward(self, sub, rel, obj=None):
        h = self.ent_emb(sub)
        r = self.rel_emb(rel)

        if obj is not None:
            t = self.ent_emb(obj)
            score = torch.sum(h * r * t, dim=1) + self.bias[obj]
            return torch.sigmoid(score)
        else:
            # 1-N Scoring
            # (B, D) * (B, D) -> (B, D)
            hr = h * r 
            # (B, D) @ (D, N) -> (B, N)
            score = torch.mm(hr, self.ent_emb.weight.transpose(1, 0))
            score += self.bias.expand_as(score)
            return torch.sigmoid(score)


class ComplEx(BaseKGE):
    def __init__(self, args):
        super(ComplEx, self).__init__(args)
        # ComplEx uses Complex Embeddings: Real + Imag
        # Standard implementation uses dim for Real and dim for Imag => Total params 2*dim
        # We assume args.emb_dim is the size of EACH component (Real/Imag)
        self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim * 2)
        
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def forward(self, sub, rel, obj=None):
        # h: (B, 2D) -> re, im
        h_all = self.ent_emb(sub)
        r_all = self.rel_emb(rel)
        
        h_re, h_im = torch.chunk(h_all, 2, dim=-1)
        r_re, r_im = torch.chunk(r_all, 2, dim=-1)
        
        if obj is not None:
            # Pairwise
            t_all = self.ent_emb(obj)
            t_re, t_im = torch.chunk(t_all, 2, dim=-1)
            
            # Score = <h,r,t> = Re(<h,r,bar(t)>)
            # Expansion: (hr_re * tr_re + hr_im * tr_im) + ...
            # Standard ComplEx Score: <Re(h), Re(r), Re(t)> + <Re(h), Im(r), Im(t)> + <Im(h), Re(r), Im(t)> - <Im(h), Im(r), Re(t)>
            
            score = torch.sum(h_re * r_re * t_re, 1) + \
                    torch.sum(h_re * r_im * t_im, 1) + \
                    torch.sum(h_im * r_re * t_im, 1) - \
                    torch.sum(h_im * r_im * t_re, 1)
            return torch.sigmoid(score)
        else:
            # 1-N Scoring
            # Precompute query terms
            # Term 1: (h_re * r_re) - (h_im * r_im)   [Matches with t_re]
            # Term 2: (h_re * r_im) + (h_im * r_re)   [Matches with t_im]
            
            # But standard formula components:
            # 1. h_re * r_re * t_re
            # 2. h_re * r_im * t_im
            # 3. h_im * r_re * t_im
            # 4. - h_im * r_im * t_re
            
            # Grouping by T component:
            # T_re interacts with: (h_re * r_re) - (h_im * r_im)
            # T_im interacts with: (h_re * r_im) + (h_im * r_re)
            
            q_re = (h_re * r_re) - (h_im * r_im)
            q_im = (h_re * r_im) + (h_im * r_re)
            
            # All Candidates
            t_all = self.ent_emb.weight
            t_re, t_im = torch.chunk(t_all, 2, dim=-1)
            
            score = torch.mm(q_re, t_re.transpose(1, 0)) + \
                    torch.mm(q_im, t_im.transpose(1, 0))
            
            return torch.sigmoid(score)


class ConvE(BaseKGE):
    def __init__(self, args):
        super(ConvE, self).__init__(args)
        # ConvE Hyperparams (Hardcoded reasonable defaults if not in args)
        # k_h, k_w for reshaping. 100 dim -> 10x10 is common
        # 200 dim -> 10x20
        self.emb_dim = args.emb_dim
        
        # Auto-detect reshape dimensions
        if self.emb_dim == 200:
            self.k_h = 10
            self.k_w = 20
        elif self.emb_dim == 100:
            self.k_h = 10
            self.k_w = 10
        else:
            # Fallback square root
            self.k_h = int(np.sqrt(self.emb_dim))
            self.k_w = self.emb_dim // self.k_h
            
        self.ker_sz = 3
        self.num_filt = 32
        
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_filt)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        
        self.drop_1 = nn.Dropout(0.2)
        self.drop_2 = nn.Dropout(0.3)
        self.feature_drop = nn.Dropout(0.2)
        
        self.m_conv1 = nn.Conv2d(1, out_channels=self.num_filt, kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=False)
        
        flat_sz_h = int(2*self.k_w) - self.ker_sz + 1
        flat_sz_w = self.k_h - self.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        
        self.fc = nn.Linear(self.flat_sz, self.emb_dim)
        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def forward(self, sub, rel, obj=None):
        h = self.ent_emb(sub).view(-1, 1, self.k_w, self.k_h)
        r = self.rel_emb(rel).view(-1, 1, self.k_w, self.k_h)
        
        # Stack h and r: (B, 1, 2*W, H) ? Or (B, 1, H, 2*W)?
        # Standard: Stack along dimension 2 (Width?)
        # Let's align with HoGRN ConvE implementation details if available, 
        # or standard PyG/LibKGE.
        # Standard: Stack embedding [h; r] -> reshape
        
        # Let's flatten first then stack 2D
        h_flat = self.ent_emb(sub).view(-1, 1, self.emb_dim)
        r_flat = self.rel_emb(rel).view(-1, 1, self.emb_dim)
        
        # Stack 1D then Reshape: (N, 2, D) -> (N, 1, 2*H, W) is weird.
        # Standard ConvE: reshape(h) -> (N, 1, 10, 10), reshape(r) -> (N, 1, 10, 10)
        # Concat along dimension 2: (N, 1, 20, 10)
        stacked_inputs = torch.cat([h, r], 2) # (B, 1, 2*k_w, k_h)
        
        x = self.bn0(stacked_inputs)
        x = self.m_conv1(x) # (B, Filt, H', W')
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.drop_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # x is the query vector (B, D)
        
        if obj is not None:
            t = self.ent_emb(obj)
            score = torch.sum(x * t, dim=1) + self.bias[obj]
            return torch.sigmoid(score)
        else:
            # (B, D) @ (D, N)
            score = torch.mm(x, self.ent_emb.weight.transpose(1, 0))
            score += self.bias.expand_as(score)
            return torch.sigmoid(score)

class TuckER(BaseKGE):
    def __init__(self, args):
        super(TuckER, self).__init__(args)
        # TuckER Parameters
        # W: Core Tensor (D_r, D_e, D_e)
        
        self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.emb_dim, self.emb_dim, self.emb_dim)), 
                                    dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True))
        
        self.input_dropout = nn.Dropout(0.3)
        self.hidden_dropout1 = nn.Dropout(0.4)
        self.hidden_dropout2 = nn.Dropout(0.5)
        
        self.bn0 = nn.BatchNorm1d(self.emb_dim)
        self.bn1 = nn.BatchNorm1d(self.emb_dim)

    def init_weights(self):
        super().init_weights()
        # Custom init for core tensor usually random uniform
        
    def forward(self, sub, rel, obj=None):
        h = self.ent_emb(sub)
        x = self.bn0(h)
        x = self.input_dropout(x)
        x = x.view(-1, 1, self.emb_dim) # (B, 1, D)

        r = self.rel_emb(rel)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1)) # (B, D) * (D, D*D) -> (B, D*D)
        W_mat = W_mat.view(-1, self.emb_dim, self.emb_dim) # (B, D, D)
        W_mat = self.hidden_dropout1(W_mat)
        
        # x * W_mat
        # (B, 1, D) * (B, D, D) -> (B, 1, D)
        x = torch.bmm(x, W_mat) 
        x = x.view(-1, self.emb_dim) 
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        
        if obj is not None:
            t = self.ent_emb(obj)
            score = torch.sum(x * t, dim=1) 
            return torch.sigmoid(score)
        else:
            score = torch.mm(x, self.ent_emb.weight.transpose(1, 0))
            return torch.sigmoid(score)



class RotatE(BaseKGE):
    def __init__(self, args):
        super(RotatE, self).__init__(args)
        self.gamma = args.margin
        # RotatE embeddings are complex: (Re, Im)
        # We assume emb_dim is the real dimension, so total embedding size is 2*emb_dim?
        # Alternatively, we treat emb_dim as total and split.
        # HoGRN usually defines embed_dim as total.
        # But RotatE paper usually doubles it.
        # Let's check HoGRN implementation if they have RotatE. They didn't in models.py list.
        # Common practice: ent_emb has shape (num_ent, dim*2), rel_emb has (num_rel, dim) (phases)
        
        self.embedding_range = (self.gamma + 2.0) / self.emb_dim
        self.ent_emb = nn.Embedding(self.num_ent, self.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.num_rel, self.emb_dim)
        
        nn.init.uniform_(self.ent_emb.weight.data, -self.embedding_range, self.embedding_range)
        nn.init.uniform_(self.rel_emb.weight.data, -self.embedding_range, self.embedding_range)
        
        self.gamma = args.margin

    def init_weights(self):
        # Override BaseKGE init
        pass

    def forward(self, sub, rel, obj=None):
        pi = 3.14159265358979323846
        
        h = self.ent_emb(sub) # (batch, 2*dim)
        r = self.rel_emb(rel) # (batch, dim)
        
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        
        # Make relation complex phase
        phase_relation = r / (self.embedding_range / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        # Rotate head
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        if obj is not None:
            # Training Mode (Pairwise)
            t = self.ent_emb(obj)
            re_tail, im_tail = torch.chunk(t, 2, dim=-1)
            
            re_diff = re_score - re_tail
            im_diff = im_score - im_tail
            diff = torch.cat([re_diff, im_diff], dim=1)
            dist = torch.norm(diff, p=1, dim=1)
            
            return self.gamma - dist
        else:
            # Compare with ALL tails
            # all_ents: (num_ent, 2*dim)
            all_re_tail, all_im_tail = torch.chunk(self.ent_emb.weight, 2, dim=-1) # (num_ent, dim)
            
            # re_score: (batch, dim). UNSQUEEZE -> (batch, 1, dim)
            # all_re_tail: UNSQUEEZE -> (1, num_ent, dim)
            
            re_diff = re_score.unsqueeze(1) - all_re_tail.unsqueeze(0)
            im_diff = im_score.unsqueeze(1) - all_im_tail.unsqueeze(0)
            
            # Stack and Norm
            diff = torch.cat([re_diff, im_diff], dim=2) # (batch, num_ent, 2*dim)
            dist = torch.norm(diff, p=1, dim=2) # (batch, num_ent)
            
            score = self.gamma - dist
            return torch.sigmoid(score)
