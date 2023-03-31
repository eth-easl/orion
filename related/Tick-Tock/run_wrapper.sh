datestr=$(date '+%H-%M-%S-%Y-%m-%d')

python run.py > ${datestr}_output.log 2>&1 &
disown
