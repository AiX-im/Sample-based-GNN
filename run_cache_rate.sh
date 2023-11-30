# echo -e "\n\n./nts_run.sh reddit gcn"
# ./nts_run.sh reddit gcn
# echo -e "\n\n./nts_run.sh lj_large gcn"
# ./nts_run.sh lj_large gcn
# echo -e "\n\n./nts_run.sh orkut gcn"
# ./nts_run.sh orkut gcn
# echo -e "\n\n./nts_run.sh wiki gcn"
# ./nts_run.sh wiki gcn
# echo -e "\n\n./nts_run.sh products gcn"
# ./nts_run.sh products gcn

# echo -e "\n\n./nts_run.sh reddit graphsage"
# ./nts_run.sh reddit graphsage
# echo -e "\n\n./nts_run.sh lj_large graphsage"
# ./nts_run.sh lj_large graphsage
# echo -e "\n\n./nts_run.sh orkut graphsage"
# ./nts_run.sh orkut graphsage
# echo -e "\n\n./nts_run.sh wiki graphsage"
# ./nts_run.sh wiki graphsage
# echo -e "\n\n./nts_run.sh products graphsage"
# ./nts_run.sh products graphsage

# 下面是reorder后的图
# echo -e "\n\n./nts_run.sh reddit_reorder gcn"
# ./nts_run.sh reddit_reorder gcn
# echo -e "\n\n./nts_run.sh lj_large_reorder gcn"
# ./nts_run.sh lj_large_reorder gcn
# echo -e "\n\n./nts_run.sh orkut_reorder gcn"
# ./nts_run.sh orkut_reorder gcn
# echo -e "\n\n./nts_run.sh wiki_reorder gcn"
# ./nts_run.sh wiki_reorder gcn
# echo -e "\n\n./nts_run.sh products_reorder gcn"
# ./nts_run.sh products_reorder gcn

# echo -e "\n\n./nts_run.sh reddit_reorder graphsage"
# ./nts_run.sh reddit_reorder graphsage
# echo -e "\n\n./nts_run.sh lj_large_reorder graphsage"
# ./nts_run.sh lj_large_reorder graphsage
# echo -e "\n\n./nts_run.sh orkut_reorder graphsage"
# ./nts_run.sh orkut_reorder graphsage
# echo -e "\n\n./nts_run.sh wiki_reorder graphsage"
# ./nts_run.sh wiki_reorder graphsage
# echo -e "\n\n./nts_run.sh products_reorder graphsage"
# ./nts_run.sh products_reorder graphsage


datasets=("reddit_reorder" "lj_large_reorder" "orkut_reorder" "wiki_reorder" "products_reorder")
cache_rates=(0.05 0.10 0.15 0.20 0.25)
algs=("graphsage" "gcn") #"graphsage" "gcn" "gat"
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
            old_file="${algs[i]}_${datasets[j]}_sample.cfg"
            new_file="${algs[i]}_${datasets[j]}_${cache_rates[k]}_sample.cfg"
            new_type="${datasets[j]}_${cache_rates[k]}"
            cat ${old_file} > ${new_file}
            sed -i '$d' ${new_file}
            echo "CACHE_RATE:${cache_rates[k]}" >> ${new_file}
            cat ${new_file}
            echo "./nts_run.sh ${new_type} ${algs[i]}"
            ./nts_run.sh ${new_type} ${algs[i]}
            # echo "run: ./run.sh ${datasets[j]} ${hidden_nums[j]} ${batch_sizes[i*dataset_num+j]} ${algs[i]}"
            # ./run.sh ${datasets[j]} ${hidden_nums[j]} ${batch_sizes[i*dataset_num+j]} ${algs[i]}
            rm ${new_file}
        done;
        # echo ${algs[j]};
        # echo ${batch_sizes[i*alg_num+j]}
    done;
done;

