para_num=$#
if [ $para_num -ne 2 ]
then
	echo "usage"
        echo "	$0 <dataset> <alg> "
        exit 1
fi
dataset=$1
# node_num=$2
# hidden_num=$3
# batch_size=$4
alg=$2
# feature_len=0
# label_num=0
# if [ $para_num -eq 7 ]; then
#         feature_len=$6
#         label_num=$7
# fi
cfg_file="${alg}_${dataset}_sample.cfg"
run_dir=$(pwd)
# 需要修改数据集的基本路径
# GRAPH_ROOT="/home/toao/文档/数据集"
log_dir="${dataset}_${alg}"
FILENAME="./log/${log_dir}"
echo ${FILENAME}
./cpu.sh &
cpu_pid=$!
./gpu.sh &
gpu_pid=$!
# random 生成的命令
# python3.8 -u 4_gpu_only_train.py --graph-dir /home/toao/文档/数据集/lj-large --node-num 7489073 --hidden-num 64 --batch-size 500 --alg gat --random true --feature-len 50 --label-num 3

# 使用原始feature的命令
# python3.8 -u 4_gpu_only_train.py --graph-dir /home/toao/文档/数据集/reddit --node-num 232965 --hidden-num 256 --batch-size 50 --alg gat
# graph_dir="${GRAPH_ROOT}/${dataset}"
# if [ $para_num -eq 5 ]; then
#         python3.8 ./4_gpu_only_train.py --graph-dir ${graph_dir} --node-num ${node_num} --hidden-num ${hidden_num} --batch-size ${batch_size} --alg ${alg}  > ./output.log
# else
#         python3.8 ./4_gpu_only_train.py --graph-dir ${graph_dir} --node-num ${node_num} --hidden-num ${hidden_num} --batch-size ${batch_size} --alg ${alg} --random --feature-len ${feature_len} --label-num ${label_num} > ./output.log
# fi

# 运行nts
echo "./build/nts ${cfg_file} > ./output.log"
./build/nts ${cfg_file} > ./output.log

# PID=$(ps -e|grep top|awk '{printf $1}')
kill -9 ${cpu_pid}
# PID=$(ps -e|grep nvidia-smi|awk '{printf $1}')
kill -9 ${gpu_pid}

sleep 1s
rm -r ${FILENAME}
mkdir -p ${FILENAME}
mv ./cpu_sgnn.log ${FILENAME}
mv ./nvidia_sgnn.log ${FILENAME}
mv  ./output.log ${FILENAME}
cp ${cfg_file} ${FILENAME}
cd ${FILENAME}
awk '{print $1","$9,","$NF}' ./cpu_sgnn.log > ./cpu_two.csv
# 本地是12，服务器是13
awk '{print $(NF-3)","$NF}' ./nvidia_sgnn.log > ./gpu_rate.csv
start_time=$(tail -n 3 output.log | head -n 1)
end_time=$(tail -n 2 output.log | head -n 1)
cd ${run_dir}
python3.8 get_rate.py ${FILENAME} ${start_time} ${end_time}

cd ${FILENAME}
tail -n 20 output.log >> ./rate_result.txt
cat ./rate_result.txt
cd ${run_dir}


