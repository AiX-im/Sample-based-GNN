para_num=$#
if [ $para_num -ne 2 ]
then
	echo "usage"
        echo "	$0 <dataset> <alg> "
        exit 1
fi
dataset=$1
alg=$2
cfg_file="${alg}_${dataset}_sample.cfg"
run_dir=$(pwd)

log_dir="${dataset}_${alg}"
FILENAME="./log/${log_dir}"
echo ${FILENAME}
./cpu.sh &
cpu_pid=$!
./gpu.sh &
gpu_pid=$!

# run
echo "./build/nts ${cfg_file} > ./output.log"
./build/nts ${cfg_file} > ./output.log

kill -9 ${cpu_pid}
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
awk '{print $(NF-3)","$NF}' ./nvidia_sgnn.log > ./gpu_rate.csv
start_time=$(tail -n 3 output.log | head -n 1)
end_time=$(tail -n 2 output.log | head -n 1)
cd ${run_dir}
python3.8 get_rate.py ${FILENAME} ${start_time} ${end_time}

cd ${FILENAME}
tail -n 20 output.log >> ./rate_result.txt
cat ./rate_result.txt
cd ${run_dir}