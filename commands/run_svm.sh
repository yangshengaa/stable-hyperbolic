# specify parameters

# select dataset: whatever is available in data/svm/
dataset='cifar fashion-mnist'

# select model types
# choices = [LSVMPP ESVM LSVM PSVM]
models="LSVMPP"

# other parameters
device='cuda'
lr_candidates="1e-7"
decision_mode="arcsinh"
tag='real'
epochs=10000
C=5

for data in $dataset; do 
    for lr in $lr_candidates; do 
        for model in $models; do 

            if [ $model == 'LSVM' ]; 
            then
                penalty="l1"
            else
                penalty="l2"
            fi

            python src/train_svm.py \
                --model $model \
                --data $data \
                --penalty $penalty \
                --loss squared_hinge \
                --C $C \
                --lr $lr \
                --epochs $epochs \
                --refpt raw \
                --decision-mode $decision_mode \
                --tag $tag \
                --device $device 
        done 
    done 
done 