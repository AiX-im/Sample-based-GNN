datasets=("products_reorder")

cache_rates=(0.10 0.12)
gpu_nums=(1 2 4 8) # 1 2 4 8
gpu_envs=("0" "1,2" "1,3,5,7" "0,1,2,3,4,5,6,7")
algs=("gcn") #"graphsage" "gcn" "gat"
models=("GCNSAMPLEPCMULTI")
batch_sizes=(1024 512) # 512 1024
# echo ${#batch_sizes[@]}
# for(( i=0;i<${#batch_sizes[@]};i++)) 
# #${#array[@]}获取数组长度用于循环
# do
#     echo ${batch_sizes[i]};
# done;
dataset_num=${#datasets[@]}
alg_num=${#algs[@]}
for(( i=0;i<${#algs[@]};i++)) 
#${#array[@]}获取数组长度用于循环
do
    # echo ${datasets[i]};
    for(( j=0;j<${#datasets[@]};j++)) 
    #${#array[@]}获取数组长度用于循环
    do

        for(( k=0;k<${#cache_rates[@]};k++))
        do
            for(( l=0;l<${#models[@]};l++))
            do
                for(( n=0;n<${#batch_sizes[@]};n++))
                do
                    for(( m=0;m<${#gpu_nums[@]};m++))
                    do

                        old_file="${algs[i]}_${datasets[j]}_sample.cfg"
                        new_file="${algs[i]}_${datasets[j]}_${cache_rates[k]}_${models[l]}_sample.cfg"
                        new_type="${datasets[j]}_${cache_rates[k]}_${models[l]}"
                        c_rate=${cache_rates[k]}
                        g_num=${gpu_nums[m]}
                        # cache_rate=`expr $c_rate / $g_num`
                        cache_rate=`echo "scale=4; $c_rate/$g_num" | bc`
                        echo "${c_rate} ${g_num} cache rate: ${cache_rate}"
                        cat ${old_file} > ${new_file}
                        sed -i '$d' ${new_file}
                        sed -i '$d' ${new_file}
                        sed -i '$d' ${new_file}
                        sed -i '1d' ${new_file}
                        echo "ALGORITHM:${models[l]}" >> ${new_file}
                        echo "CACHE_RATE:0${cache_rate}" >> ${new_file}
                        echo "GPU_NUM:${gpu_nums[m]}" >> ${new_file}
                        echo "BATCH_SIZE:${batch_sizes[n]}" >> ${new_file}
                        cat ${new_file}
                        echo "CUDA_VISIBLE_DEVICES=${gpu_envs[m]} ./nts_run_multi.sh ${new_type} ${algs[i]} ${log_dir}"
                        log_dir="${datasets[j]}_bs${batch_sizes[n]}_${models[l]}_c${cache_rates[k]}_g${gpu_nums[m]}"
                        echo "log dir: ${log_dir}"

                        CUDA_VISIBLE_DEVICES=${gpu_envs[m]} ./nts_run_multi.sh ${new_type} ${algs[i]} ${log_dir}

                        # echo "run: ./run.sh ${datasets[j]} ${hidden_nums[j]} ${batch_sizes[i*dataset_num+j]} ${algs[i]}"
                        # ./run.sh ${datasets[j]} ${hidden_nums[j]} ${batch_sizes[i*dataset_num+j]} ${algs[i]}
                        rm ${new_file}

                        echo -e "\n\n"
                    done
                done
            done;
        done;
        # echo ${algs[j]};
        # echo ${batch_sizes[i*alg_num+j]}
    done;
done;

