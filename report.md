# `.eval()` mode for torch model 
This is a report on the effectiveness of putting the torch model into eval mode. The goal of this report is to determine if the model is more deterministic in eval mode at inference time (no training involved).

## What is eval mode? 
A model can switch to "eval" mode by calling `model.eval()` or `model.train(False)`. Official documentation of the mode is at https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval but it is not very clear since the its effect is module-specific. 

## Methodology

For this test, I ran 4 different videos through the SWT model, 3 times with training mode and 3 times with eval mode. 
Sourcing mmifs: In order to obtain these mmifs, I created the base mmif for each of these mp4’s using clams source. After that, I used the containerized image of the SWT v7.5 with default settings to obtain the results of these files 3 times. In order to replicate this in the eval model, after adding the self.classifier.eval() line below the line https://github.com/clamsproject/app-swt-detection/blob/8c46437d52f7c49d472821c08b23cfba0a6f7411/modeling/classify.py#L33 
, I built the app as a podman image that I again ran with the default settings. These results are in [this directory](https://github.com/BenLambright/swt-determinism-eval/tree/main/data).

### Calculations:   
To get the average variance in annotations, I just used a simple percent difference formula: $\frac{|a-b| + |a-c| + |b-c|}{3}$, where $a$, $b$, and $c$ are the counts of annotations from each trial.

To get all other values, I calculated the [Jaccard coefficient](https://en.wikipedia.org/wiki/Jaccard_index): $\frac{a \cap b \cap c}{a \cup  b \cup c}$.

The pythonic versions of these calculations can be found in `calc_variance.py` below.

Metrics:  
Average variance in annotations: variance the exact number of annotations per swt output (see formula above)
Jaccard coefficient of time points: variance in the timepoints. This should always stay the same because I never altered the sample rate, but I wanted to confirm with this calculation. 
Jaccard coefficient of labels: variance in the labels
Jaccard coefficient of tuple counts: tuple pairing each timepoint to its corresponding label, measures the variance in these tuples, so the how variance in how each timepoint is assigned to a label.

## Results

(Jaccard coefficients in percent)  

* `train=True`
    - Average variance in annotations: 1.167  
    - Jaccard coefficient of time points: 1.0  
    - Jaccard coefficient of label counts: 0.972  
    - Jaccard coefficient of tuple counts: 0.78

* `train=False`
    - Average variance in annotations: 0.833  
    - Jaccard coefficient of time points: 1.0  
    - Jaccard coefficient of label counts: 0.984  
    - Jaccard coefficient of tuple counts: 0.903

## Discussion

The average variance in label counts was slightly improved, and the average variance in annotations and variance in the time points were effectively the same. The big takeaway is that the tuple counts were significantly less variable in eval mode (12.3 percent). This suggests that there is generally an even distribution of labels and timepoints in the swt files, but the model doesn’t always predict a given timepoint to have the same label. We therefore much prefer using the model in eval mode in order to decrease the variance.


## Code 

<details>

<summary>calc_variance.py</summary>

``` python
from collections import Counter
import os
import json

def get_file_counts(file_dict):
    """Get the total count of annotations in a file, a count of all timepoints, counts of all labels, and a tuple of each timepoint and its label"""
    annotation_count = 0
    timepoints = Counter()
    label_counter = Counter()
    tuples = Counter()  # timepoints and labels

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
                tuples[(annotation_property["timePoint"], annotation_property["label"])] += 1

    # print(f"file annotation count: {annotation_count}")
    # print(f"file time points: {timepoints}")
    # print(f"file label count: {label_counter}")
    # print(f"file tuple count: {tuples}")
    # print()

    return annotation_count, timepoints, label_counter, tuples

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
    tuple_difference = jaccard_similarity(file1[3], file2[3], file3[3])

    print(f"annotation count average difference: {annotation_count_difference}")
    print(f"timepoints variation: {timepoint_difference}")
    print(f"label count variation: {label_difference}")
    print(f"tuple count variation: {tuple_difference}")

    return annotation_count_difference, timepoint_difference, label_difference, tuple_difference

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
        outfile.write(f"tuple count: {results[3]}\n")
```
  
</details>
