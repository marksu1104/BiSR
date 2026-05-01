import os
import subprocess
from pathlib import Path

# 準備要連續執行的目標資料集
datasets = ["NELL23K", "WD-singer", "WN18RR", "FB15K-237-10", "FB15K-237-20", "FB15K-237-50"]

# 定義每個資料集在 HoGRN (ConvE 或是 DistMult) 的最佳官方參數設定
# 根據 baselines/HoGRN/sh 內部的腳本提取
configs = {
    "NELL23K": "-rel_reason -pre_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func conve -chan_drop 0.1 -rel_mask 0.1 -rel_norm -hid_drop 0.3",
    "WD-singer": "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5",
    "WN18RR": "-rel_reason -reason_type 'mixdrop2' -bias -batch 256 -init_dim 200 -gcn_dim 200 -embed_dim 200 -gcn_layer 1 -gcn_drop 0.0 -score_func conve -chamix_dim 300 -relmix_dim 300 -rel_norm -hid_drop 0.3 -hid_drop2 0.5 -feat_drop 0.1 -k_w 10 -k_h 20 -num_filt 250 -ker_sz 7",
    "FB15K-237-10": "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5",
    "FB15K-237-20": "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.2 -score_func distmult -chamix_dim 400 -relmix_dim 400 -rel_norm -hid_drop 0.3",
    "FB15K-237-50": "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.2 -score_func distmult -chamix_dim 400 -relmix_dim 400 -rel_norm -hid_drop 0.3"
}

def main():
    # 因為腳本已經搬到 HoGRN 底下，我們可以直接確認當前目錄並執行
    target_dir = Path(".")
    
    # 也可以透過檢查 run.py 是不是在當前目錄來防呆
    if not (target_dir / "run.py").exists():
        print(f"錯誤: 找不到 run.py，請確認是否在 HoGRN 目錄下執行此腳本")
        return

    # 不再需要切換工作目錄了
    # os.chdir(target_dir)

    # 開始遍歷 6 個資料集並執行
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"🚀 開始執行 HoGRN 訓練與驗證: {ds}")
        print(f"{'='*60}")
        
        args = configs.get(ds, configs["FB15K-237-10"]) # 如果沒有就拿 FB15K 的當預設兜底
        
        # 組裝執行指令： CUDA_VISIBLE_DEVICES=0 python run.py -data {ds} {args}
        cmd = f"CUDA_VISIBLE_DEVICES=0 python run.py -data {ds} {args}"
        
        print(f"執行的 Command: {cmd}\n")
        
        # 開啟 log 儲存，以便我們後補擷取 (導向 stderr 到 stdout，一起存)
        log_file = f"HoGRN_{ds}.log"
        full_cmd = f"{cmd} 2>&1 | tee {log_file}"
        
        try:
            # 加上 stdout 導向，讓使用者能清楚看見運作過程
            subprocess.run(full_cmd, shell=True, check=True)
            print(f"\n✅ {ds} 執行完成！")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {ds} 執行失敗，錯誤碼: {e.returncode}")

if __name__ == '__main__':
    main()
    
    # 執行所有實驗後，自動進行結果整理
    print("\n📦 開始整理 HoGRN 測試結果...")
    try:
        subprocess.run("python aggregate_hogrn_results.py", shell=True, check=True)
    except Exception as e:
        print(f"整理結果失敗: {e}")

    main()