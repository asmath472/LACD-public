import json
from collections import Counter

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    # File paths
    file_paths = ["./data/datasets/LACD-biclassification/train-test-divide/"+p for p in ["test.jsonl", "train.jsonl", "val.jsonl"]]

    # Set to hold unique elements
    unique_elements = set()

    # Process each file to extract the specified elements
    for file_path in file_paths:
        data = read_jsonl(file_path)
        for row in data:
            if "article1" in row and "source_law_id" in row and "source_article" in row:
                unique_elements.add((row["article1"], row["source_law_id"], row["source_article"]))
            if "article2" in row and "target_law_id" in row and "target_article" in row:
                unique_elements.add((row["article2"], row["target_law_id"], row["target_article"]))

    # Count the distribution of law_id
    law_id_counter = Counter(element[1] for element in unique_elements)


    # Calculate the threshold for "Others" grouping (5% of total)
    total_count = sum(law_id_counter.values())
    threshold = 0.01 * total_count

    # Separate items into "Others" if they fall below the threshold
    filtered_counter = Counter()
    others_count = 0

    for law_id, count in law_id_counter.items():
        if count >= threshold:
            filtered_counter[law_id] = count
        else:
            others_count += count

    # Add "Others" to the counter if there are any
    if others_count > 0:
        filtered_counter["Others"] = others_count

    # Sort by count in descending order
    sorted_law_id_distribution = filtered_counter.most_common()

    # Print the sorted distribution of law_id
    print("\nDistribution of law_id (with 'Others' for less frequent items):")
    for law_id, count in sorted_law_id_distribution:
        print(f"  {law_id}: {count}")

    print(len(law_id_counter))