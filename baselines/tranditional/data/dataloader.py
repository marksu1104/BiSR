import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class KGData:
    def __init__(self, data_path, add_inverse=True):
        self.data_path = data_path
        self.add_inverse = add_inverse
        self.ent2id = {}
        self.rel2id = {}
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        
        self.load_data()
        
        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id)
        
        # Group definitions for 1-N training
        self.train_sr2o = defaultdict(set)
        self.all_sr2o = defaultdict(set)
        
        self.process_1N_data()
        
    def load_data(self):
        # First pass: Collect entities and relations
        ents = set()
        rels = set()
        
        for split in ['train', 'valid', 'test']:
            path = os.path.join(self.data_path, f"{split}.txt")
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    # Checked dataset format (FB15K-237, WN18RR in this workspace): Entity1, Entity2, Relation
                    # h, t, r
                    h, t, r = parts[0], parts[1], parts[2]
                    ents.add(h)
                    ents.add(t)
                    rels.add(r)
        
        # Create mappings
        self.ent2id = {e: i for i, e in enumerate(sorted(list(ents)))}
        self.rel2id = {r: i for i, r in enumerate(sorted(list(rels)))}
        
        if self.add_inverse:
            num_base_rels = len(self.rel2id)
            for r, i in list(self.rel2id.items()):
                self.rel2id[r + "_reverse"] = i + num_base_rels
                
        # Second pass: Convert to IDs
        num_base_rels = len(rels)
        
        for split in ['train', 'valid', 'test']:
            path = os.path.join(self.data_path, f"{split}.txt")
            if not os.path.exists(path): continue
            
            triples = []
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    # h, t, r format
                    h, t, r = parts[0], parts[1], parts[2]
                    h_id, r_id, t_id = self.ent2id[h], self.rel2id[r], self.ent2id[t]
                    
                    triples.append((h_id, r_id, t_id))
                    
            if split == 'train':
                self.train_triples = triples
                # Add inverse to train triples ONLY if we want to train reverse relations too
                if self.add_inverse:
                     inv_triples = []
                     for h, r, t in triples:
                         inv_triples.append((t, r + num_base_rels, h))
                     self.train_triples.extend(inv_triples)
            elif split == 'valid':
                self.valid_triples = triples
            elif split == 'test':
                self.test_triples = triples

    def process_1N_data(self):
        # Build sr2o maps
        # For training: only train triples (including inverses if added)
        for h, r, t in self.train_triples:
            self.train_sr2o[(h, r)].add(t)
            self.all_sr2o[(h, r)].add(t)
            
        # For validation/test: Add to all_sr2o (for filtering)
        # We also need inverse mappings for filtering if we do tail prediction on inverse relations (which is head prediction)
        num_base = len(self.rel2id) // 2 if self.add_inverse else len(self.rel2id)
        
        for split_triples in [self.valid_triples, self.test_triples]:
            for h, r, t in split_triples:
                self.all_sr2o[(h, r)].add(t)
                if self.add_inverse:
                    inv_r = r + num_base
                    self.all_sr2o[(t, inv_r)].add(h)

class TrainDatasetOriginal(Dataset):
    """
    Original Pairwise Training Dataset (1-vs-1 Negative Sampling)
    """
    def __init__(self, triples, num_ent, num_neg=1):
        self.triples = triples
        self.num_ent = num_ent
        self.num_neg = num_neg
        
    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Negative Sampling: Corrupt Head or Tail (50% probability)
        if np.random.rand() < 0.5:
             # Corrupt Head
             neg_h = np.random.randint(0, self.num_ent)
             while neg_h == h:
                 neg_h = np.random.randint(0, self.num_ent)
             return torch.tensor([h, r, t], dtype=torch.long), torch.tensor([neg_h, r, t], dtype=torch.long)
        else:
             # Corrupt Tail
             neg_t = np.random.randint(0, self.num_ent)
             while neg_t == t:
                 neg_t = np.random.randint(0, self.num_ent)
             return torch.tensor([h, r, t], dtype=torch.long), torch.tensor([h, r, neg_t], dtype=torch.long)


class TrainDataset(Dataset):
    def __init__(self, data_dict, num_ent, label_smoothing=0.1):
        self.data = list(data_dict.keys()) # (h, r) keys
        self.sr2o = data_dict
        self.num_ent = num_ent
        self.label_smoothing = label_smoothing
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        h, r = self.data[idx]
        train_objects = list(self.sr2o[(h, r)])
        
        # Multi-hot Label
        label = torch.zeros(self.num_ent)
        label[train_objects] = 1.0
        
        if self.label_smoothing > 0:
            label = (1.0 - self.label_smoothing) * label + (1.0 / self.num_ent)
            
        return torch.tensor([h, r]), label

class EvalDataset(Dataset):
    def __init__(self, triples, all_sr2o, num_ent):
        self.triples = triples
        self.all_sr2o = all_sr2o
        self.num_ent = num_ent
    
    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Construct multi-hot label of ALL true tails for this (h, r) query
        # This is used for "Filtered Setting" evaluation
        true_tails = self.all_sr2o.get((h, r), set())
        
        label = torch.zeros(self.num_ent)
        if true_tails:
            # list conversion required for indexing
            label[list(true_tails)] = 1.0
            
        return torch.tensor([h, r, t]), label

def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
