# DuoDR: Dual-Stream Collaborative Contrastive Learning with Dual-Axis Path Consistency for Drug Repositioning

Motivation: Drug repositioning aims to find new disease indications for existing drugs, offering a faster and more cost-effective alternative to developing new drugs from scratch. However, traditional approaches still face several challenges, including the mismatch between similarity and interaction profiles, hub bias in neighbor-based evidence, and insufficient inference support in predictions.  

Results: We propose a two-stage framework for drug repositioning, named DuoDR, which integrates a dual-stream collaborative contrastive learning module (DS-CCL) with a dual-axis neighborhood-aware refinement module (DA-NAR). DS-CCL aligns chemical/phenotypic and topological embeddings by using an information noise contrastive estimation. DA-NAR performs bidirectional inferences with an adaptive evidence-weighting strategy to alleviate hub bias and rescue low-score predictions. Experiments on four benchmark datasets show that DuoDR achieves strong performance, and especially DA-NAR further improves the performance for ranking.  

Conclusions: This work provides an alternative algorithm for drug repositioning, helpful for accelerating the identification of new potential drug-disease relationships.  


## Requirements

```bash
Pytorch >= 1.6
dgl==0.6.0post1
python==3.8.1
```


## Codes 
#drug_train.py: training script for DuoDR.  <br>
#tslr_inference.py: inference script with DA-NAR refinement on the prediction matrix.  <br>
#data.py / dataDenovo.py: data loaders for drug-disease associations and similarity matrices.  <br>
#model.py / layers.py: model definition and neural network layers.  <br>
#evaluate.py: evaluation utilities (AUROC/AUPR and related metrics).  <br>


## Dataset
A dataset is located in the directory: DuoDR\raw_data

This dataset includes: 
1. drug-disease associations;  
2. drug-drug similarity matrix and disease-disease similarity matrix.

## Train
To train the model, run the following command:

```bash
python drug_train.py --data_name lrssl --num_neighbor 8 --folds 10
```

## Inference
Inference needs the training output directory:

```bash
python tslr_inference.py --data_name lrssl --save_dir neighbor_num8/lrssl_1time
```

## Cite
If you use DuoDR in your research, please cite: Xu et al. DuoDR: Dual-Stream Collaborative Contrastive Learning with Dual-Axis Neighborhood-Aware Refinement for Drug Repositioning


## Contact<br>
Email: xiang.ju@foxmail.com
