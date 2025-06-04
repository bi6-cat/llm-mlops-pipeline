import pandas as pd
import argparse
import sys

def check_csv(path: str):
    # Kiểm tra một file CSV:
    # 1) Phải có cột 'text' và 'label'.
    # 2) Không có dòng 'text' trống (blank hoặc NaN).
    # 3) Nhãn 'label' chỉ thuộc {0, 1}.
    # Trả về True nếu hợp lệ, False nếu có lỗi.

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Không thể đọc file {path}: {e}")
        return False

    # 1. Kiểm cột 'text' và 'label'
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"[ERROR] File {path} thiếu cột: {missing}")
        return False

    # 2. Kiểm text rỗng hoặc NaN
    empty_text = df["text"].isna().sum() + (df["text"].str.strip() == "").sum()
    if empty_text > 0:
        print(f"[ERROR] File {path} có {empty_text} dòng text rỗng hoặc NaN")
        return False

    # 3. Nhãn ngoài {0,1}
    unique_labels = set(df["label"].unique().tolist())
    invalid_labels = unique_labels - {0, 1}
    if invalid_labels:
        print(f"[ERROR] File {path} có nhãn không hợp lệ: {invalid_labels}")
        return False

    # 4. Thông tin thêm: số dòng
    print(f"[INFO] File {path} hợp lệ, số dòng: {len(df)}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate IMDb CSV data")
    parser.add_argument("--train", type=str, required=True, help="Đường dẫn train.csv")
    parser.add_argument("--test", type=str, required=True, help="Đường dẫn test.csv")
    args = parser.parse_args()

    ok_train = check_csv(args.train)
    ok_test  = check_csv(args.test)
    if not (ok_train and ok_test):
        print("[ERROR] Data validation failed.")
        sys.exit(1)

    print("[INFO] Data validation passed.")
    sys.exit(0)
