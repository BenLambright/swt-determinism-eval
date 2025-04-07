from collections import Counter
import os
import json

def get_file_counts(file_dict):
    """Get the total count of annotations in a file, a count of all timepoints, counts of all labels, and a tuple of each timepoint and its label"""
    annotation_count = 0
    timepoints = Counter()
    label_counter = Counter()
    bigrams = Counter()  # timepoints and labels

    for value in file_dict["views"]:
        view = dict(value)
        annotations = view["annotations"]
        # print(annotations)
        # print()
        for raw_annotation in annotations:
            annotation = dict(raw_annotation)
            # print(annotation["properties"])
            annotation_count += 1
            annotation_property = dict(annotation["properties"])
            if "timePoint" in annotation_property:
                timepoints[annotation_property["timePoint"]] += 1
                label_counter[annotation_property["label"]] += 1
                bigrams[(annotation_property["timePoint"], annotation_property["label"])] += 1

    # print(f"file annotation count: {annotation_count}")
    # print(f"file time points: {timepoints}")
    # print(f"file label count: {label_counter}")
    # print(f"file bigram count: {bigrams}")
    print()

    return annotation_count, timepoints, label_counter, bigrams

def jaccard_similarity(set1, set2, set3):
    """Calculating the jaccard similarity between the sets of Counters"""
    intersection = set1 & set2 & set3
    union = set1 | set2 | set3
    return sum(intersection.values()) / sum(union.values()) if union else 0.0

def open_file(file):
    """Opening file"""
    with open(file, 'r') as f:
        return json.load(f)

def average_difference(a, b, c):
    """Average pairwise difference formula"""
    ab = abs(a - b)
    ac = abs(a - c)
    bc = abs(b - c)
    return (ab + ac + bc) / 3

def compare_files(file1, file2, file3):
    """Comparing the different datapoints from each file with average difference or jaccard similarity"""
    # calculate the average difference between the annotation counts
    annotation_count_difference = average_difference(file1[0], file2[0], file3[0])

    # calculate the jaccard similarities
    timepoint_difference = jaccard_similarity(file1[1], file2[1], file3[1])
    label_difference = jaccard_similarity(file1[2], file2[2], file3[2])
    bigram_difference = jaccard_similarity(file1[3], file2[3], file3[3])

    print(f"annotation count average difference: {annotation_count_difference}")
    print(f"timepoints variation: {timepoint_difference}")
    print(f"label count variation: {label_difference}")
    print(f"bigram count variation: {bigram_difference}")

    return annotation_count_difference, timepoint_difference, label_difference, bigram_difference

if __name__=="__main__":
    data_dir = "f551104e446/eval"
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files]

    data_for_file = []
    for file in files:
        file_dict = open_file(file)
        data_for_file.append(get_file_counts(file_dict))

    results = compare_files(data_for_file[0], data_for_file[1], data_for_file[2])

    with open(f"{data_dir}_variance.txt", 'w') as outfile:
        outfile.write(f"annotation count: {results[0]}\n")
        outfile.write(f"time points: {results[1]}\n")
        outfile.write(f"label count: {results[2]}\n")
        outfile.write(f"bigram count: {results[3]}\n")

