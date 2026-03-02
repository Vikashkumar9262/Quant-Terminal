import pandas as pd
import os

# 1. Define Paths (Absolute paths prevent "0 byte" errors)
BASE_DIR = r"C:\python\mini-gpt\backend"
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "19-02-2025-to-19-02-2026.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "input.txt")

print(f"📂 Looking for CSV at: {CSV_PATH}")

try:
    # 2. Load the Financial Data
    df = pd.read_csv(CSV_PATH)
    
    # 3. Convert Data to Text (This defines the missing variable)
    # We combine all columns into a single string for the Mini-GPT to learn from
    final_text_data = df.to_string(index=False)
    
    # 4. Save to input.txt
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(final_text_data)
        f.flush()
        os.fsync(f.fileno())

    actual_size = os.path.getsize(OUTPUT_PATH)
    print(f"✅ Success! {OUTPUT_PATH} now contains {len(df)} rows ({actual_size} bytes).")

except FileNotFoundError:
    print(f"❌ Error: Could not find the CSV file. Check if it's in {CSV_PATH}")
except Exception as e:
    print(f"❌ Error: {e}")