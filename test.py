from config import PROJECT_DIR, DATA_DIR, IMAGE_DIR, LABEL_DIR

print(f"Project directory: {PROJECT_DIR}")
print(f"Dataset directory: {DATA_DIR}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Label directory: {LABEL_DIR}")

print(f"\nDataset exists: {DATA_DIR.exists()}")
print(f"Images exist: {IMAGE_DIR.exists()}")
print(f"Labels exist: {LABEL_DIR.exists()}")