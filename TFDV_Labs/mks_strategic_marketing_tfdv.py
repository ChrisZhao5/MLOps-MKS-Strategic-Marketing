import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def generate_mks_data():
    print("Step 1: 正在生成 MKS B2B 客户数据...")
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'revenue': np.random.lognormal(3, 1.5, n),
        'days_since_last_purchase': np.random.randint(1, 400, n),
        'equipment_age': np.random.uniform(0.5, 12, n),
        'industry': np.random.choice(['Semiconductor', 'Photonics', 'Electronics'], n),
        'buy_upgrade': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })
    return df

def validate_mks_data(df):
    print("\nStep 2: 正在进行数据质检 (Data Quality Check)...")
    errors = []
    # 检查负数营收
    if (df['revenue'] < 0).any():
        errors.append("⚠️ 发现负数营收数据！")
    # 检查行业范围
    valid_industries = ['Semiconductor', 'Photonics', 'Electronics']
    if not df['industry'].isin(valid_industries).all():
        errors.append("⚠️ 发现非目标行业类别！")
    
    if errors:
        for e in errors: print(e)
    else:
        print("✅ 数据质量检查通过。")

def train_mks_model(df):
    print("\nStep 3: 正在训练 MKS 购买意向模型...")
    # 简单的特征处理
    X = df[['revenue', 'days_since_last_purchase', 'equipment_age']].values
    y = df['buy_upgrade'].values
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, verbose=0)
    print("✅ 模型训练完成。")

if __name__ == "__main__":
    # 运行流程
    mks_df = generate_mks_data()
    # 注入一个错误用来展示
    mks_df.loc[0, 'revenue'] = -100 
    
    validate_mks_data(mks_df)
    train_mks_model(mks_df)