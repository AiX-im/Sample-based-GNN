# nvidia-smi -lms 500 |grep "MiB | "| tee -a > nvidia_sgnn.log
function addTimestamp(){
    timestamp=$[$(date +%s%N)/1000000]
    # logline=$1
    local line
    # IFS=$'\n'
    while read line; do
        # logline=${line}$'\t'${timestamp}
        # echo "timestamp: ${timestamp}"
        # echo "line: ${line}"
        len=${#line}
        # echo "长度${len}"
        if [ $len -gt 10 ];then
            logline=${line}"\t"${timestamp}
            # result=`echo $logline | tr -d '\n'`
            echo -e "${logline}"
        fi
        # printf '%s\n' "$logline"
    done
}
filename=./nvidia_sgnn.log
: > ${filename}

while true
do
    nvidia-smi | grep "MiB | "| addTimestamp | tee -a >> ${filename}
    sleep 0.5s
done

