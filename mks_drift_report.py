import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def check_mks_drift():
    print("--- MKS Strategic Marketing: Model Monitoring Service ---")
    
    # 1. 模拟训练时的参考数据 (Reference)
    ref = pd.DataFrame({'equip_age': np.random.normal(5, 2, 100)})
    
    # 2. 模拟生产环境的实时数据 (Current) - 故意制造偏移
    curr = pd.DataFrame({'equip_age': np.random.normal(12, 3, 100)})
    
    # 3. 执行 KS Test (计算两个分布是否一致)
    # statistic 是差异大小, pvalue < 0.05 则认为有显著漂移
    stat, p_value = ks_2samp(ref['equip_age'], curr['equip_age'])
    
    print(f"Feature: Equipment_Age")
    print(f"Drift Statistic: {stat:.4f}")
    print(f"P-Value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("🚨 ALERT: Data Drift Detected! The market environment has changed.")
        print("Action Required: Trigger Model Retraining Pipeline.")
    else:
        print("✅ No significant drift detected. Model performance is stable.")

if __name__ == "__main__":
    check_mks_drift()