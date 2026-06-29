import pandas as pd
import glob
import os
import time
import pyarrow.parquet as pq
import pyarrow as pa

def csv_to_parquet_safe():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "data")
    
    print(f"⚡ Memory-Safe Conversion running in: {csv_dir}")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    # We already successfully built these, let's skip them to save time
    completed = ['POS_CASH_balance', 'application_train', 'bureau', 'credit_card_balance', 'application_test']
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        if base_name in completed or base_name == "HomeCredit_columns_description":
            continue
            
        output_path = os.path.join(csv_dir, f"{base_name}.parquet")
        start_time = time.time()
        print(f"Streaming {file_name} in safe chunks...")
        
        # Stream the CSV in chunks of 100k rows so RAM never spikes
        chunksize = 100_000
        parquet_writer = None
        
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize, low_memory=False)):
            # Convert the pandas chunk to an isolated PyArrow table
            table = pa.Table.from_pandas(chunk)
            
            # Initialize writer on the first chunk with schema design
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            
            parquet_writer.write_table(table)
            
        if parquet_writer:
            parquet_writer.close()
            
        elapsed = time.time() - start_time
        print(f"  Successfully Saved: {base_name}.parquet ({elapsed:.2f}s)")

if __name__ == "__main__":
    csv_to_parquet_safe()