import numpy
import pandas


def shuffle_data(data: pandas.DataFrame):
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def split_data(data: pandas.DataFrame, minor_split: float):
    split_index = int(len(data) * minor_split)

    return data.iloc[split_index:], data.iloc[:split_index]


# Load the data.
df = pandas.read_csv("raw_absorption.csv")

# Drop supplementary columns.
df_reduced = df.drop(columns=["Group", "Name"])

# Create a dataset with only non-film wool.
dataset1 = df_reduced[df_reduced["mark"] == 0].copy()

# Create a dataset with only film wool.
dataset2 = df_reduced[df_reduced["mark"].isin([1, 2])].copy()

# Rename 2 to 1 in the only film wool dataset.
dataset2["mark"] = dataset2["mark"].replace(2, 1)

# Reset the indices.
dataset1.reset_index(drop=True, inplace=True)
dataset2.reset_index(drop=True, inplace=True)

# Create a true test set.
dataset1_shuffled = shuffle_data(dataset1)
dataset2_shuffled = shuffle_data(dataset2)

split_index_1 = int(len(dataset1_shuffled) * 0.1)
split_index_2 = int(len(dataset2_shuffled) * 0.1)

dataset1_90 = dataset1_shuffled.iloc[split_index_1:]
dataset1_10 = dataset1_shuffled.iloc[:split_index_1]
dataset2_90 = dataset2_shuffled.iloc[split_index_2:]
dataset2_10 = dataset2_shuffled.iloc[:split_index_2]

test_set = pandas.concat([dataset1_10, dataset2_10], ignore_index=True)
test_set.reset_index(drop=True, inplace=True)

# Split the non-film into 5 parts.
splits = numpy.array_split(dataset1_90, 5)

# Assign to separate variables.
dataset1_part1 = splits[0]
dataset1_part2 = splits[1]
dataset1_part3 = splits[2]
dataset1_part4 = splits[3]
dataset1_part5 = splits[4]

# Split the film into 5 parts.
splits = numpy.array_split(dataset2_90, 5)

# Assign to separate variables.
dataset2_part1 = splits[0]
dataset2_part2 = splits[1]
dataset2_part3 = splits[2]
dataset2_part4 = splits[3]
dataset2_part5 = splits[4]

# Put together five-fold cross-validation sets.
ffcv1 = pandas.concat([dataset1_part1, dataset2_part1, dataset2_part2, dataset2_part3, dataset2_part4,
                       dataset2_part5], ignore_index=True)
ffcv1.reset_index(drop=True, inplace=True)
ffcv2 = pandas.concat([dataset1_part2, dataset2_part1, dataset2_part2, dataset2_part3, dataset2_part4,
                       dataset2_part5], ignore_index=True)
ffcv2.reset_index(drop=True, inplace=True)
ffcv3 = pandas.concat([dataset1_part3, dataset2_part1, dataset2_part2, dataset2_part3, dataset2_part4,
                       dataset2_part5], ignore_index=True)
ffcv3.reset_index(drop=True, inplace=True)
ffcv4 = pandas.concat([dataset1_part4, dataset2_part1, dataset2_part2, dataset2_part3, dataset2_part4,
                       dataset2_part5], ignore_index=True)
ffcv4.reset_index(drop=True, inplace=True)
ffcv5 = pandas.concat([dataset1_part5, dataset2_part1, dataset2_part2, dataset2_part3, dataset2_part4,
                       dataset2_part5], ignore_index=True)
ffcv5.reset_index(drop=True, inplace=True)

# Create five-fold cross-validation datasets.
ffcv1.to_csv('FFCV1.csv', index=False)
ffcv2.to_csv('FFCV2.csv', index=False)
ffcv3.to_csv('FFCV3.csv', index=False)
ffcv4.to_csv('FFCV4.csv', index=False)
ffcv5.to_csv('FFCV5.csv', index=False)

# Create the test set.
test_set.to_csv('TEST.csv', index=False)
