import argparse
import os
import pickle
import torch


def merge(args):
    num_files = args.num_files
    importance = []
    for i in range(num_files):

        if hasattr(args, "task_name"): 
            input_filename = "importance_{}_".format(args.task_name) + str(i) + ".pkl"
        else:
            input_filename = "importance_" + str(i) + ".pkl"

        with open(input_filename, "rb") as file:
            data = pickle.load(file)
            importance.append(data)
    importance = torch.tensor(importance)
    importance = importance.sum(0)
    importance_sorted, importance_idx = torch.sort(importance, dim=1, descending=True)
    print("Done merging...")

    result = {
        "importance": importance_sorted.tolist(),
        "idx": importance_idx.tolist()
    }

    if hasattr(args, "task_name"): 
        output_filename="importance_{}.pkl".format(args.task_name)
    else :
        output_filename="importance.pkl" 

    with open(output_filename, "wb") as file:
        pickle.dump(result, file)
    print("Done dumping...")

    for i in range(num_files):

        if hasattr(args, "task_name"): 
            input_filename = "importance_{}_".format(args.task_name) + str(i) + ".pkl"
        else:
            input_filename = "importance_" + str(i) + ".pkl"

        os.remove(input_filename)
    print("Done removing files...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, 
                        help="Task Name")
    parser.add_argument("--num_files", type=int, default=1,
                        help="Number of files to merge.")
    args = parser.parse_args()

    merge(args)
    print("Completed!")


if __name__ == "__main__":
    main()
