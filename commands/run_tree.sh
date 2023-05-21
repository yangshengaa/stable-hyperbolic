# specify parameters

# select model 
# choices = ["TreeEuclidean", "TreeLorentz", "TreePoincare", "TreeEuclideanLiftLorentz"]
model="TreeEuclideanLiftLorentz"  

# select tree 
# choices are ones available in data/tree/
dataset="401 403"

# other parameters
obj_candidates="scaled"
lr_candidates="10 100"
opt="sgd"
lift_type="expmap"

for data in $dataset; do 
    for obj in $obj_candidates; do 
        for lr in $lr_candidates; do 

            if [ $lr == "10" ]; then 
                epochs=100000
            elif [ $lr == "100" ]; then 
                epochs=50000
            fi

            PYTHONDONTWRITEBYTECODE=1 python src/train_tree.py \
                --data "sim_tree_${data}" \
                --model $model \
                --lift-type $lift_type \
                --opt $opt \
                --obj $obj \
                --lr $lr \
                --epochs $epochs \
                --log-train \
                --log-train-epochs 2000
        done 
    done 
done 