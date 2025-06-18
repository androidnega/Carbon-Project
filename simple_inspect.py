import csv

print("=== DATASET INSPECTION (Simple CSV Reader) ===")

# Read CSV file manually
with open('data/Carbon Emission.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # Get header
    header = next(reader)
    print(f"Columns ({len(header)}): {header}")
    
    # Read first few rows
    rows = []
    for i, row in enumerate(reader):
        if i < 5:  # First 5 rows
            rows.append(row)
        else:
            break
    
    print(f"\nFirst {len(rows)} rows:")
    for i, row in enumerate(rows):
        print(f"Row {i+1}: {row}")
    
    # Count total rows
    file.seek(0)
    next(reader)  # Skip header
    total_rows = sum(1 for row in reader)
    print(f"\nTotal data rows: {total_rows}")

print("\n=== COLUMN ANALYSIS ===")
for i, col in enumerate(header):
    print(f"{i+1}. {col}")
