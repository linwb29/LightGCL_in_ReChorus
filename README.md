## Introduction
本项目旨在基于ReChorus框架实现LightGCL模型的复现。
LightGCL模型是一种基于图神经网络的推荐系统模型，其核心思想是将用户与物品之间的交互关系建模为图，并通过轻量化的图卷积网络学习用户和物品的嵌入表示，从而进行精准的推荐。与传统的图神经网络方法不同，LightGCL通过简化图卷积操作，采用了更加高效的邻接矩阵传播策略，显著提升了计算效率和模型的可扩展性。
LightGCL利用SVD降维增强了邻接矩阵的表示能力，使得模型能够更好地捕捉用户和物品之间的潜在关系，并在多层图卷积传播中逐步融合信息，学习到更加精细的用户和物品嵌入。通过这种方式，LightGCL不仅能在大规模数据上高效运行，还能显著提升推荐的精度，成为高效的协同过滤方法之一。
## code
本项目基于ReChorus框架实现了LightGCL模型的复现。
`LightGCL.py`文件位于`LightGCL_ReChorus/src/models/general`目录下，实现了LightGCL模型的初始化、数据编码、前向传播、损失函数计算等功能。
`LightGCLBase`类实现了LightGCL模型的基本结构，定义了嵌入维度、层数、SVD 的秩等超参数，构造了用户-物品的稀疏邻接矩阵，对邻接矩阵进行 SVD 分解。
`LightGCL`类继承自`LightGCLBase`类，封装了完整的 LightGCL 模型, 包括模型初始化、前向传播等功能。
`LightGCLEncoder`实现嵌入的初始化和传播逻辑（多层图卷积）。
可以直接下载本项目的`ReChorus`文件夹，并运行我们下面给出的代码，即可复现LightGCL模型。也可以只下载`LightGCL.py`文件，将其放在下载好的ReChorus框架中（具体位置为`ReChorus/src/models/general`），然后运行我们下面给出的代码。
## requirement
```
python == 3.10.4
torch == 2.5.0+cu118
numpy == 1.22.0
ipython == 8.10.0
jupyter == 1.0.0
tqdm == 4.66.1
pandas == 1.4.4
scikit-learn == 1.1.3
scipy == 1.7.3
```
## training
由于硬件条件受限，我们仅在cpu环境下进行训练。如果有gpu环境，可以将 `--gpu`后面的参数设为0，使用gpu进行训练。
训练的数据位于 `ReChorus/data`，我们提供两个数据集进行TOP-K训练任务，分别是`Grocery_and_Gourmet_Food`和`MIND_Large/MINDTOPK`。输出放在`ReChorus/log`目录下。
下面是我们对不同模型的训练命令：
```cd src```<br>
对于`Grocery_and_Gourmet_Food`数据集：<br>
LightGCL:```python main.py --gpu -1 \ --model_name LightGCL \ --emb_size 64 \ --n_layers 3 \ --svd_rank 5 \ --lr 1e-3 \ --l2 1e-8 \ --dataset 'Grocery_and_Gourmet_Food'```<br>
LightGCN:```python main.py --gpu -1 --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset 'Grocery_and_Gourmet_Food'```<br>
BPRMF:```python main.py --gpu -1 --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'```<br>
对于`MIND_Large/MINDTOPK`数据集：<br>
LightGCL:```python main.py --gpu -1 \ --model_name LightGCL \ --emb_size 64 \ --n_layers 3 \ --svd_rank 5 \ --lr 1e-3 \ --l2 1e-8 \ --dataset 'MIND_Large\MINDTOPK' ```<br>
LightGCN:```python main.py --gpu -1 --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset MIND_Large\MINDTOPK```<br>
BPRMF:```python main.py --gpu -1 --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MIND_Large\MINDTOPK```<br>
所有训练均在Windows环境下进行，训练时间较长，请耐心等待。
## parameter
可能要用到的参数：
gpu $\Rightarrow$ gpu的编号: 使用该编号的gpu进行训练
model_name $\Rightarrow$ 模型名称: 使用该模型进行训练
dataset $\Rightarrow$ 数据集名称: 使用该数据集进行训练
lr $\Rightarrow$ 学习率: 学习率
l2 $\Rightarrow$ l2正则化系数: l2正则化系数
n_layers $\Rightarrow$ 图神经网络层数: 图神经网络层数
svd_rank $\Rightarrow$ 矩阵分解时保留的特征个数: 仅在LightGCL模型中使用

---------------------------------------------------------
ReChorus框架代码地址：```https://github.com/THUwangcy/ReChorus```
如果在运行代码过程中遇到什么问题，请在企业微信上联系我们。