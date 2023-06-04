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
filename=./cpu_sgnn.log
: > ${filename}
while true
do
    top -b -n 1 | grep -w nts | addTimestamp | tee -a >> ${filename}
    # top -b -n 1 | grep code | tee -a >> ./cpu_utilization.log
    sleep 0.5s
done