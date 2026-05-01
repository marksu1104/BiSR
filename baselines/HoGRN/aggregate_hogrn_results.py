import os
import re
import pandas as pd
import glob

def aggregate_logs():
    log_files = glob.glob('HoGRN_*.log')
    results = []

    # Regex to match the HoGRN output format:
    # EVAL_STD model=HoGRN split=test dir=avg filtered=1 tie_aware=1 full_entity=1 mrr={:.5f} h1={:.5f} h3={:.5f} h10={:.5f}
    pattern = re.compile(
        r"EVAL_STD model=.*? split=test dir=avg filtered=1 tie_aware=1 full_entity=1 mrr=([\d.]+) h1=([\d.]+) h3=([\d.]+) h10=([\d.]+)"
    )

    for log_file in log_files:
        dataset = log_file.replace('HoGRN_', '').replace('.log', '')
        
        mrr, h1, h3, h10 = None, None, None, None
        
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    mrr = float(match.group(1))
                    h1 = float(match.group(2))
                    h3 = float(match.group(3))
                    h10 = float(match.group(4))
        
        if mrr is not None:
            results.append({
                'Dataset': dataset,
                'Model': 'HoGRN',
                'MRR': mrr,
                'Hits@1': h1,
                'Hits@3': h3,
                'Hits@10': h10
            })
            print(f"找到 {dataset} 的測試結果")
        else:
            print(f"警告：找不到 {dataset} 的測試結果 (可能執行失敗或尚未完成)")

    if results:
        df = pd.DataFrame(results)
        # 整理欄位順序
        df = df[['Dataset', 'Model', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']]
        df.to_csv('hogrn_baselines_summary.csv', index=False)
        print("\n✅ 所有 HoGRN 結果已匯總至 baselines/HoGRN/hogrn_baselines_summary.csv")
    else:
        print("\n❌ 沒有找到任何 HoGRN 測試結果可以整理")

if __name__ == "__main__":
    aggregate_logs()
