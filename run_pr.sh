#!/bin/bash

# can append parameter values in name for readability
model_name=model_
result_name=result_
max_acc=-1
max_model=0

for i in 1 2 3 4 5 6 7 8 9 10
do
    python train_pr.py -i adultdtrain-sensitivelast -o $model_name$i 2>/dev/null
    python predict_lr.py -i adultdtest-sensitivelast -o $result_name$i -m $model_name$i 2>/dev/null

done

echo "$result_name\n\n"
for i in 1 2 3 4 5 6 7 8 9 10
do
    echo "result #$i"
    python fai_bin_bin.py $result_name$i | grep "### Acc" -A 1
done
