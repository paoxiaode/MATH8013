acts=(0 1 2 3 4)
freqs=(1 2 4 8)
dims=(1 10 100 300)
name=result_$(date +%H_%M_%S)

# Different activation func
for act in ${acts[@]};
do
    echo run $act
    # original model
    python main.py --num_layers 5 --hidden_dim 300 --act $act --dir ${name}
done

# Different frequency
for freq in ${freqs[@]};
do
    echo run $freq
    # original model
    python main.py --num_layers 5 --hidden_dim 300 --act 2 --dir ${name} --fre $freq
done

# Different dimension
for dim in ${dims[@]};
do
    echo run $dim
    # original model
    python main.py --num_layers 5 --hidden_dim $dim --act 2 --dir ${name} 
done