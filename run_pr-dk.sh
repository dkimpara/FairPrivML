#!/bin/bash

### edit the following variables --- make sure to change the model_name and result_name
###   variables between runs

# enter command line param args to be used: e.g. -e 100
params=''

# append parameter values in name for readability: e.g. model_eta100_
model_name=model_
result_name=result_



for i in 1 2 3 4 5 6 7 8 9 10
do
    python train_pr.py -i adultdtrain-sensitivelast -o $model_name$i $params 2>/dev/null
    python predict_lr.py -i adultdtest-sensitivelast -o $result_name$i -m $model_name$i 2>/dev/null

done



printf "$result_name\n\n"

# maxacc=0
# maxmodel=0
#
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#     #echo "result #$i"
#     python fai_bin_bin.py $result_name$i | grep "### Acc" -A 1
#     s=$(python fai_bin_bin.py result_$i | awk '/Acc/ { getline; print }')
#     a=( $s )
#     v=${a[5]}
#     acc=${v::-1}
#     if ((  $(echo "$acc > $maxacc" | bc -l)  ))
#     then
#      maxacc=$acc
#      maxmodel=$i
#     fi
# done

# for i in 1 2 3 4 5 6 7 8 9 10
# do
#  if [ "$i" -ne "$maxmodel" ]
#   then
#    rm $model_name$i
#    rm $result_name$i
#   fi
# done


# 
# printf "\n\n\n"
# echo "Highest accuracy of $maxacc in $model_name$maxmodel"
# printf "\n\n\n"
