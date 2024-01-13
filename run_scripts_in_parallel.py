import subprocess
import multiprocessing

cwd = "/home/kimth/workspace/MoEBERT"

def run_bash_script(cmd):
    subprocess.run(args=cmd, cwd=cwd)

def main():
    list_task_name = ["cola", "mnli", "mrpc", "qnli", "sst2"]
    list_mode = ["dense", "importance", "moe"]

    cmd_format = "bash bert_base_classification.sh {} {} {} {}"

    for mode in list_mode:

        # set num workers
        if mode == "importance":
            num_workers = 1
        else:
            num_workers = 4

        commands = []
        for idx, task_name in enumerate(list_task_name):
            cuda_device = idx % 2
            port_num = 9000 + idx

            cmd = cmd_format.format(task_name, cuda_device, port_num, mode)
            cmd_list = cmd.split(" ")
            commands.append(cmd_list)

        # Run commands in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(run_bash_script, commands)

if __name__ == "__main__":
    main()
