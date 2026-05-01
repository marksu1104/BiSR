import os
import glob
import pandas as pd
import re

LOG_DIR = "logs"

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Directory {LOG_DIR} does not exist. No logs to parse.")
        return

    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    
    results = []

    # Regex to capture metrics from the line:
    # [Epoch 0 test]: MRR: 0.15579, Hits@1: 0.090666, Hits@3: 0.16854, Hits@10: 0.2974
    eval_pattern = re.compile(r"test\]: MRR:\s*([\d\.]+),\s*Hits@1:\s*([\d\.]+),\s*Hits@3:\s*([\d\.]+),\s*Hits@10:\s*([\d\.]+)")

    for log_file in log_files:
        filename = os.path.basename(log_file)
        # Typically filename looks like TransE_NELL23K.log
        if not filename.endswith(".log"):
            continue
            
        base_name = filename[:-4]
        # Some models or datasets might have underscores in them, so let's carefully split
        # We know models are typically like TransE, DistMult, ComplEx, ConvE, TuckER, RotatE
        # By default our script names it MODEL_DATASET.log -> split by first '_'
        parts = base_name.split("_", 1)
        if len(parts) == 2:
            model_name, dataset = parts
        else:
            model_name, dataset = base_name, "Unknown"

        mrr_val, h1_val, h3_val, h10_val = None, None, None, None

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # We want the LAST occurrence of test evaluation
            for line in reversed(lines):
                match = eval_pattern.search(line)
                if match:
                    mrr_val = float(match.group(1))
                    h1_val  = float(match.group(2))
                    h3_val  = float(match.group(3))
                    h10_val = float(match.group(4))
                    break # Found the last test evaluation

        results.append({
            "Dataset": dataset,
            "Model": model_name,
            "MRR": mrr_val,
            "Hits@1": h1_val,
            "Hits@3": h3_val,
            "Hits@10": h10_val
        })

    if not results:
        print("No valid results found in logs.")
        return

    df = pd.DataFrame(results)
    
    # Sort the table for readability
    df = df.sort_values(by=["Dataset", "Model"])
    
    # Print the table
    print("\n" + "="*80)
    print(" Traditional Baselines Final Evaluation Results ")
    print("="*80)
    print(df.to_string(index=False, na_rep="N/A"))
    print("="*80 + "\n")
    
    # Save to CSV
    out_csv = "traditional_baselines_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Summary saved to {out_csv}")

if __name__ == "__main__":
    main()
