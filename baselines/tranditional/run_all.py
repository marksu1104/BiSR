import os
import subprocess
import argparse
import sys

# Configuration (aligned with run_all_baselines.sh)
MODELS = ["TransE", "DistMult", "ComplEx", "ConvE", "TuckER", "RotatE"]
DATASET = "FB15K-237-10"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable

DEFAULT_ARGS = {
    "max_epochs": 500,
    "emb_dim": 200,
    "batch_size": 256,
    "lr": 1e-3,
    "gpu": 0,
    "patience": 25,
    "eval_freq": 1,
}

def get_dataset_path(dataset_name):
    return os.path.abspath(os.path.join(BASE_DIR, f"../../datasets/{dataset_name}"))


def run_with_tee(cmd, log_file):
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

def run_experiment(model, dataset, dry_run=False, **kwargs):
    data_path = get_dataset_path(dataset)
    if not os.path.exists(data_path):
        print(f"Skipping {dataset}: Path {data_path} not found.")
        return

    cmd = [PYTHON_EXEC, "-u", "main.py"]

    args = DEFAULT_ARGS.copy()
    args.update(kwargs)

    cmd.extend(["--model", model])
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--data_path", data_path])

    for k, v in args.items():
        cmd.extend([f"--{k}", str(v)])

    if model in {"TransE", "RotatE"}:
        cmd.extend(["--margin", "9.0"])

    print(f"\n[{'DRY RUN' if dry_run else 'RUNNING'}] Model: {model} | Dataset: {dataset}")
    print(f"Command: {' '.join(cmd)}")

    if not dry_run:
        try:
            os.makedirs("logs", exist_ok=True)
            log_file = f"logs/{model}_{dataset}.log"
            run_with_tee(cmd, log_file)

        except subprocess.CalledProcessError as e:
            print(f"Error running {model} on {dataset}: {e}")
        except KeyboardInterrupt:
            print("Interrupted by user.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Experiments (aligned with run_all_baselines.sh)")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="Models to include")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset name")
    parser.add_argument("--gpu", type=int, default=DEFAULT_ARGS['gpu'], help="GPU ID")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_ARGS['max_epochs'], help="Epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_ARGS['batch_size'], help="Batch size")
    parser.add_argument("--patience", type=int, default=DEFAULT_ARGS['patience'], help="Early stopping patience")
    parser.add_argument("--emb_dim", type=int, default=DEFAULT_ARGS['emb_dim'], help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=DEFAULT_ARGS['lr'], help="Learning rate")
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_ARGS['eval_freq'], help="Evaluation frequency")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")

    args = parser.parse_args()

    print(f"Starting Baseline Experiments for {args.dataset}...")
    print("Logs will be saved to 'logs/' directory and displayed in terminal.")
    print(f"Unified Parameters: Emb_Dim={args.emb_dim}, LR={args.lr}, Batch={args.batch_size}")

    for model in args.models:
        print("-" * 48)
        print(f"Running {model}...")
        run_experiment(
            model,
            args.dataset,
            dry_run=args.dry_run,
            gpu=args.gpu,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            emb_dim=args.emb_dim,
            lr=args.lr,
            eval_freq=args.eval_freq,
        )

    print("All baseline experiments completed. Check logs directory for results.")

if __name__ == "__main__":
    main()
