import subprocess
import multiprocessing

cwd = "/home/kimth/workspace/MoEBERT"

def run_bash_script(cmd):
    print("Run {}".format(cmd))
    subprocess.run(args=cmd, cwd=cwd)

def main():
    list_mode = [
        # "dense",
        # "importance",
        # "moe",
        "diffmoe"
    ]
    list_task_name = [
        "rte",
        #"cola",
        #"mrpc",
        # "sst2",
        # "qnli",
        # "mnli",
        # "qqp",
        ]

    list_moebert_expert_num=[
        4,
        8,
        16
    ]
    list_moebert_expert_dim=[
        3072
    ]
    list_moebert_share_importance=[
        #0,
        1024,
        2048,
        3072
    ]
    list_moebert_target_sparsity=[
        0.1,
        0.05,
        0.01,
    ]

    cmd_format = "bash bert_base_classification.sh {} {} {} {} {} {} {} {}"

    run_id = 0
    for idx1, mode in enumerate(list_mode):

        # set num workers
        if mode == "importance":
            num_workers = 1
        else:
            num_workers = 2

        for idx2, task_name in enumerate(list_task_name):
            commands = []
            for moebert_expert_num in list_moebert_expert_num:
                for moebert_expert_dim in list_moebert_expert_dim:
                    for moebert_share_importance in list_moebert_share_importance:
                        for moebert_target_sparsity in list_moebert_target_sparsity:

                            cuda_device = run_id % 2
                            #cuda_device = 1

                            # print("Running in CUDA{}".format(cuda_device))

                            port_num = 9000 + run_id
                            run_id += 1

                            cmd = cmd_format.format(
                                task_name,
                                cuda_device,
                                port_num,
                                mode,
                                moebert_expert_num,
                                moebert_expert_dim,
                                moebert_share_importance,
                                moebert_target_sparsity,
                            )
                            cmd_list = cmd.split(" ")
                            commands.append(cmd_list)
                            #commands.append(cmd)

                            # Run this loop only once if not diffmoe
                            if mode != "diffmoe" :
                                break

            # Run commands in parallel
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.map(run_bash_script, commands, chunksize=1)
                #res = pool.imap(run_bash_script, commands)
            #[print(commands)


if __name__ == "__main__":
    main()
