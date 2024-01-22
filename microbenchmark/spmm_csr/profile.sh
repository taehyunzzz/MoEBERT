export CUDA_VISIBLE_DEVICES=0
result_dir="prof_results"
mkdir -p ${result_dir}

sparsity=0.99

for dmodel in 768; do
    dff=$(( ${dmodel} * 4 ))
    # for num_tokens in 1 4 16 64 256 1024; do 
    for num_tokens in 1 16 64 256 1024; do 

        result_file="${result_dir}/prof_dff${dff}_dmodel${dmodel}_token${num_tokens}_sparsity${sparsity}.nsys-rep"
        CMD="
            ./spmm_csr_example
            -m ${dff}
            -k ${dmodel}
            -n ${num_tokens}
            -s ${sparsity}
        "
        NSYS_CMD="
            nsys profile
            --output=${result_file}
            --force-overwrite=true
            --trace=cuda,osrt,nvtx
            --cudabacktrace=true
            ${CMD}
        "
        ${CMD}

        sleep 1s;

    done
done