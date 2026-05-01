
import argparse
import torch
import os
from main import Trainer

def evaluate_test(args):
    # Initialize Trainer (loads data and builds model structure)
    trainer = Trainer(args)
    
    # Load Best Model
    ckpt_path = os.path.join(trainer.ckpt_dir, f"best_model_{args.model}_{args.dataset}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    print(f"Loading checkpoint from {ckpt_path}...")
    trainer.model.load_state_dict(torch.load(ckpt_path, map_location=trainer.device))
    
    # Evaluate on Test
    print("Evaluating on Test Set...")
    metrics = trainer.evaluate(split='test', epoch=0)
    print("Test Set Results:")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../datasets/FB15K-237")
    parser.add_argument("--dataset", type=str, default="FB15K-237")
    parser.add_argument("--model", type=str, default="TransE")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--margin", type=float, default=9.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    # Dummy args required by Trainer init
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num_neg", type=int, default=1)
    parser.add_argument("--loss_margin", type=float, default=1.0)
    
    args = parser.parse_args()
    evaluate_test(args)
