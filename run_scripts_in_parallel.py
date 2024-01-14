import subprocess
import multiprocessing

cwd = "/home/kimth/workspace/MoEBERT"

def run_bash_script(cmd):
    subprocess.run(args=cmd, cwd=cwd)

def main():
    list_mode = [
        # "dense",
        # "importance",
        "moe"
    ]
    list_task_name = [
        "rte",
        # "cola",
        # "mrpc",
        # "sst2",
        # "qnli",
        # "mnli",
        # "qqp",
        ]

    cmd_format = "bash bert_base_classification.sh {} {} {} {}"

    for idx1, mode in enumerate(list_mode):

        # set num workers
        if mode == "importance":
            num_workers = 1
        else:
            num_workers = 4

        commands = []
        for idx2, task_name in enumerate(list_task_name):
            cuda_device = idx2 % 2
            # cuda_device = 1
            port_num = 9200 + idx1 * len(list_task_name) + idx2

            cmd = cmd_format.format(task_name, cuda_device, port_num, mode)
            cmd_list = cmd.split(" ")
            commands.append(cmd_list)

        # Run commands in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(run_bash_script, commands)

if __name__ == "__main__":
    main()
