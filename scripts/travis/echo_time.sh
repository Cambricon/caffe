#? /bin/bash
count=0
while [ $count -le 20 ]
do
  date
  sleep 1m
  count=`expr $count + 1`
done
