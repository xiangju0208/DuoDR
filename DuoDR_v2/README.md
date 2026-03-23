## Environment Setup

```bash
Pytorch >= 1.6
dgl==0.6.0post1
python==3.8.1
```

Activate the environment:

```bash
conda activate duodr
```

## Running

### 1) Training (10-fold)

```bash
python drug_train.py --data_name lrssl --num_neighbor 8 --folds 10
```

### 2) Inference (Baseline + P-TSLR)

Inference needs the training output directory and fold id:

```bash
python tslr_inference.py --data_name lrssl --save_dir neighbor_num8/lrssl_1time --save_id 1
```