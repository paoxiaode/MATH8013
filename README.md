# MATH8013小作业

**the-waves@sjtu.edu.cn**

## Code structure

```
├── log                    #运行结果目录
├── main.py                #运行主函数
├── Network                #模型代码
│   ├── DNN.py
│   ├── __init__.py
├── README.md
├── run.sh                 #测试脚本
├── result                 #原始代码
└── util                   #画图、计算等函数 
    ├── folder.py
    ├── functions.py
    ├── __init__.py
    ├── parameter.py
    ├── plot.py
    └── utils.py
```

## How to run

``` python
# bash
source run.sh

# --act 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
python main.py --num_layers 5 --hidden_dim 300 --act 0
```