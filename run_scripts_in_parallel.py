import subprocess
import multiprocessing

cwd = "/home/kimth/workspace/MoEBERT"

def run_bash_script(cmd):
    # print("Run {}".format(" ".join(cmd)))
    subprocess.run(args=cmd, cwd=cwd)

list_task_name = [
    "rte",
    "cola",
    "mrpc",
    # "sst2",
    # "qnli",
    # "mnli",
    # "qqp",
    ]

def sweep_expert():
    list_mode = [
        # "dense",
        # "importance",
        "moe",
        "diffmoe"
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
        # 0,
        # 1024,
        2048,
        # 3072
    ]
    list_moebert_target_sparsity=[
        # 0.1,
        # 0.05,
        0.01,
        # 0.005,
        # 0
    ]

    return {
        "list_mode"                     :list_mode,
        "list_moebert_expert_num"       :list_moebert_expert_num,
        "list_moebert_expert_dim"       :list_moebert_expert_dim,
        "list_moebert_share_importance" :list_moebert_share_importance,
        "list_moebert_target_sparsity"  :list_moebert_target_sparsity,
    }

def sweep_shared_dim():
    list_mode = [
        # "dense",
        # "importance",
        "moe",
        "diffmoe"
    ]
    list_moebert_expert_num=[
        # 4,
        8,
        # 16
    ]
    list_moebert_expert_dim=[
        3072
    ]
    list_moebert_share_importance=[
        0,
        1024,
        2048,
        3072
    ]
    list_moebert_target_sparsity=[
        # 0.1,
        # 0.05,
        0.01,
        # 0.005,
        # 0
    ]

    return {
        "list_mode"                     :list_mode,
        "list_moebert_expert_num"       :list_moebert_expert_num,
        "list_moebert_expert_dim"       :list_moebert_expert_dim,
        "list_moebert_share_importance" :list_moebert_share_importance,
        "list_moebert_target_sparsity"  :list_moebert_target_sparsity,
    }

def sweep_target_sparsity():
    list_mode = [
        # "dense",
        # "importance",
        # "moe",
        "diffmoe"
    ]
    list_moebert_expert_num=[
        # 4,
        8,
        # 16
    ]
    list_moebert_expert_dim=[
        3072
    ]
    list_moebert_share_importance=[
        # 0,
        # 1024,
        2048,
        # 3072
    ]
    list_moebert_target_sparsity=[
        # 0.1,
        0.05,
        0.01,
        0.005,
        0
    ]

    return {
        "list_mode"                     :list_mode,
        "list_moebert_expert_num"       :list_moebert_expert_num,
        "list_moebert_expert_dim"       :list_moebert_expert_dim,
        "list_moebert_share_importance" :list_moebert_share_importance,
        "list_moebert_target_sparsity"  :list_moebert_target_sparsity,
    }

def run_sweep(list_mode,
              list_moebert_expert_num,
              list_moebert_expert_dim,
              list_moebert_share_importance,
              list_moebert_target_sparsity,
              num_base_workers=4,
              base_port_num=6000,
              ):

    cmd_format = "bash bert_base_classification.sh {} {} {} {} {} {} {} {}"

    run_id = 0
    commands = []
    for mode in list_mode:

        # set num workers
        num_workers = 1 if mode == "importance" else num_base_workers

        for task_name in list_task_name:
            for moebert_expert_num in list_moebert_expert_num:
                for moebert_expert_dim in list_moebert_expert_dim:
                    for moebert_share_importance in list_moebert_share_importance:
                        for moebert_target_sparsity in list_moebert_target_sparsity:

                            cuda_device = run_id % 2

                            port_num = base_port_num + run_id
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

                            # Run this loop only once if not diffmoe
                            if mode != "diffmoe" :
                                break

    # Run commands in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_bash_script, commands, chunksize=1)

def main():
    run_sweep( **sweep_expert()          , num_base_workers=6) 
    run_sweep( **sweep_shared_dim()      , num_base_workers=6) 
    run_sweep( **sweep_target_sparsity() , num_base_workers=6) 
    
if __name__ == "__main__":
    main()
