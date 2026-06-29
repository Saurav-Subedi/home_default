import pandas as pd
import numpy as np
import os
import time

def build_bureau_features(data_dir="./data"):
    print("⏳ Aggregating Bureau and Bureau Balance tables...")
    bureau = pd.read_parquet(os.path.join(data_dir, "bureau.parquet"))
    bb = pd.read_parquet(os.path.join(data_dir, "bureau_balance.parquet"))
    
    bureau.columns = [c.strip().upper() for c in bureau.columns]
    bb.columns = [c.strip().upper() for c in bb.columns]
    
    bb_cat = pd.get_dummies(bb, columns=['STATUS'], prefix='STATUS')
    bb_cat_cols = [col for col in bb_cat.columns if col.startswith('STATUS_')]
    
    bb_agg = bb_cat.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'count'],
        **{col: ['mean'] for col in bb_cat_cols}
    })
    bb_agg.columns = [f"BB_{col[0]}_{col[1].upper()}" for col in bb_agg.columns]
    bb_agg = bb_agg.reset_index()
    
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    bureau_cat = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_TYPE'])
    cat_cols = [c for c in bureau_cat.columns if 'CREDIT_ACTIVE_' in c or 'CREDIT_TYPE_' in c]
    
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean']
    }
    bureau_final_specs = {**num_aggregations, **{col: ['mean'] for col in cat_cols}}
    bureau_agg = bureau_cat.groupby('SK_ID_CURR').agg(bureau_final_specs)
    bureau_agg.columns = [f"BUREAU_{c[0]}_{c[1].upper()}" for c in bureau_agg.columns]
    
    print(f"✅ Bureau Features: {bureau_agg.shape[1]} metrics created.")
    return bureau_agg

def build_previous_application_features(data_dir="./data"):
    print("⏳ Aggregating Previous Applications...")
    prev = pd.read_parquet(os.path.join(data_dir, "previous_application.parquet"))
    prev.columns = [c.strip().upper() for c in prev.columns]
    
    for col in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        prev[col] = prev[col].replace({365243: np.nan})
        
    prev_cat = pd.get_dummies(prev, columns=['NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE'])
    cat_cols = [c for c in prev_cat.columns if 'NAME_CONTRACT_STATUS_' in c or 'NAME_CONTRACT_TYPE_' in c]
    
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean', 'sum'],
        'AMT_CREDIT': ['max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean', 'sum']
    }
    prev_final_specs = {**num_aggregations, **{col: ['mean'] for col in cat_cols}}
    prev_agg = prev_cat.groupby('SK_ID_CURR').agg(prev_final_specs)
    prev_agg.columns = [f"PREV_{c[0]}_{c[1].upper()}" for c in prev_agg.columns]
    
    print(f"✅ Previous App Features: {prev_agg.shape[1]} metrics created.")
    return prev_agg

def build_installment_features(data_dir="./data"):
    print("⏳ Aggregating Installment Payments via Memory-Safe ID Chunks (13M+ rows)...")
    
    # Read just the unique target IDs first to plan the chunks without loading full records
    inst_info = pd.read_parquet(os.path.join(data_dir, "installments_payments.parquet"), columns=['SK_ID_CURR'])
    unique_ids = inst_info['SK_ID_CURR'].unique()
    del inst_info # Free memory instantly
    
    chunk_size = 60_000
    aggregated_chunks = []
    
    # Process customers sequentially in safe blocks
    for i in range(0, len(unique_ids), chunk_size):
        id_chunk = unique_ids[i:i + chunk_size]
        
        # Filter read directly out of disk parquet
        inst = pd.read_parquet(
            os.path.join(data_dir, "installments_payments.parquet"),
            filters=[('SK_ID_CURR', 'in', id_chunk)]
        )
        inst.columns = [c.strip().upper() for c in inst.columns]
        
        inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
        inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
        inst['DPD'] = inst['DPD'].clip(lower=0)
        
        chunk_agg = inst.groupby('SK_ID_CURR').agg({
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
        })
        chunk_agg.columns = [f"INSTAL_{c[0]}_{c[1].upper()}" for c in chunk_agg.columns]
        aggregated_chunks.append(chunk_agg)
        print(f"   Processed installments block {min(i + chunk_size, len(unique_ids))}/{len(unique_ids)} customers...")

    inst_agg = pd.concat(aggregated_chunks, axis=0)
    print(f"✅ Installment Features: {inst_agg.shape[1]} metrics created safely.")
    return inst_agg

def main():
    DATA_DIR = "./data"
    start_pipeline = time.time()
    
    bureau_feats = build_bureau_features(DATA_DIR)
    prev_feats = build_previous_application_features(DATA_DIR)
    inst_feats = build_installment_features(DATA_DIR)
    
    print("Loading primary application profiles...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "application_train.parquet"))
    
    final_train = train.merge(bureau_feats, on='SK_ID_CURR', how='left')
    final_train = final_train.merge(prev_feats, on='SK_ID_CURR', how='left')
    final_train = final_train.merge(inst_feats, on='SK_ID_CURR', how='left')
    
    final_train['DAYS_EMPLOYED'] = final_train['DAYS_EMPLOYED'].replace({365243: np.nan})
    
    final_train['CREDIT_TO_INCOME_RATIO'] = final_train['AMT_CREDIT'] / final_train['AMT_INCOME_TOTAL']
    final_train['ANNUITY_TO_INCOME_RATIO'] = final_train['AMT_ANNUITY'] / final_train['AMT_INCOME_TOTAL']
    
    print(f"\n🚀 Completed Full Pipeline Map! Matrix Dimensions: {final_train.shape}")
    print(f"Total Execution Runtime: {time.time() - start_pipeline:.2f}s")
    
    final_train.to_parquet(os.path.join(DATA_DIR, "fe_train_final.parquet"))
    print("Saved completed feature matrix to 'fe_train_final.parquet'")

if __name__ == "__main__":
    main()