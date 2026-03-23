import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model import Net
from evaluate import evaluate
from data import DrugDataLoader
from utils import MetricLogger, common_loss, setup_seed, InfoNCELoss, random_edge_dropout, FocalLoss, mask_node_features


def train(args, dataset, graph_data, cv):
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]

    drug_graph = dataset.drug_graph.to(args.device)
    dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    args.rating_vals = dataset.possible_rel_values

    model = Net(args=args)
    model = model.to(args.device)
    rel_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(args.device)
    cl_loss_fn = InfoNCELoss(temperature=args.cl_temp).to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)
    print("Loading network finished ...\n")

    test_loss_logger = MetricLogger(['iter', 'loss', 'auroc', 'aupr'], ['%d', '%.4f', '%.4f', '%.4f'],
                                    os.path.join(args.save_dir, 'test_metric%d.csv' % args.save_id))

    train_gt_ratings = graph_data['train'][2].to(args.device)
    train_enc_graph = graph_data['train'][0].int().to(args.device)
    train_dec_graph = graph_data['train'][1].int().to(args.device)
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature
    print("Start training ...")

    start = time.perf_counter()
    best_iter, best_auroc, best_aupr = 0, 0, 0
    best_iter_aupr, best_aupr_overall, best_auroc_at_best_aupr = 0, 0, 0
    true, score = 0, 0
    for iter_idx in range(1, args.train_max_iter):
        model.train()
        Two_Stage = False

        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_proj_1, dis_proj_1, _, _ = \
            model(train_enc_graph, train_dec_graph,
                  drug_graph, drug_sim_feat, drug_feat,
                  dis_graph, dis_sim_feat, dis_feat,
                  Two_Stage)
        
        drug_graph_aug = random_edge_dropout(drug_graph, args.cl_dropout)
        dis_graph_aug = random_edge_dropout(dis_graph, args.cl_dropout)
        
        drug_sim_feat_aug = mask_node_features(drug_sim_feat, args.cl_mask_prob)
        dis_sim_feat_aug = mask_node_features(dis_sim_feat, args.cl_mask_prob)
        
        _, _, drug_sim_out_aug, _, dis_sim_out_aug, _, _, _, _ = \
            model(train_enc_graph, train_dec_graph,
                  drug_graph_aug, drug_sim_feat_aug, drug_feat,
                  dis_graph_aug, dis_sim_feat_aug, dis_feat,
                  Two_Stage)

        pred_ratings = pred_ratings.squeeze(-1)

        loss_com_drug = common_loss(drug_out, drug_sim_out)
        loss_com_dis = common_loss(dis_out, dis_sim_out)

        drug_features_cl = th.cat([drug_sim_out, drug_sim_out_aug], dim=0)
        dis_features_cl = th.cat([dis_sim_out, dis_sim_out_aug], dim=0)

        loss_cl_drug = cl_loss_fn(drug_features_cl)
        loss_cl_dis = cl_loss_fn(dis_features_cl)

        drug_features_cv = th.cat([drug_out, drug_sim_out], dim=0)
        dis_features_cv = th.cat([dis_out, dis_sim_out], dim=0)

        loss_cv_drug = cl_loss_fn(drug_features_cv)
        loss_cv_dis = cl_loss_fn(dis_features_cv)
        
        loss = rel_loss(pred_ratings, train_gt_ratings) + \
               args.beta * loss_com_dis + args.beta * loss_com_drug + \
               args.lambda_cl * (loss_cl_drug + loss_cl_dis) + \
               args.lambda_cv * (loss_cv_drug + loss_cv_dis)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        auroc, aupr, y_true, y_score = evaluate(args, model, graph_data,
                                                drug_graph, drug_feat, drug_sim_feat,
                                                dis_graph, dis_feat, dis_sim_feat)
        test_loss_logger.log(iter=iter_idx, loss=loss.item(), auroc=auroc, aupr=aupr)
        logging_str = "Iter={}, loss={:.4f}, AUROC={:.4f}, AUPR={:.4f}".format(
            iter_idx, loss.item(), auroc, aupr)

        if auroc > best_auroc:
            best_auroc = auroc
            best_aupr = aupr
            best_iter = iter_idx
            score = y_score
            true = y_true
            th.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_%d.pth' % args.save_id))

        if aupr > best_aupr_overall:
            best_aupr_overall = aupr
            best_auroc_at_best_aupr = auroc
            best_iter_aupr = iter_idx
            th.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_aupr_%d.pth' % args.save_id))

        if iter_idx % args.train_valid_interval == 0:
            print("test-logging_str", logging_str)

    result = {
        "y_score": score,
        "y_true": true
    }
    # Ensure values are lists or provide an index if scalars
    if np.isscalar(score) or np.isscalar(true):
        data_result = pd.DataFrame(result, index=[0])
    else:
        data_result = pd.DataFrame(result)

    data_result.to_csv(os.path.join(args.save_dir, '%d_result.csv' % int(cv + 1)), index=False)

    end = time.perf_counter()

    print(
        f"Fold={args.save_id} "
        f"Best(AUROC) Iter={best_iter} AUROC={best_auroc:.4f} AUPR={best_aupr:.4f} | "
        f"Best(AUPR) Iter={best_iter_aupr} AUROC={best_auroc_at_best_aupr:.4f} AUPR={best_aupr_overall:.4f}"
    )
    print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end - start))))
    test_loss_logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaDR')
    parser.add_argument('--seed', default=125, type=int)
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--model_activation', type=str, default="tanh")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_units', type=int, default=840)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--train_max_iter', type=int, default=4000) # 4000
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=100) # 100
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--nhid1', type=int, default=500)
    parser.add_argument('--nhid2', type=int, default=75)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--share_param', default=True, action='store_true')

    parser.add_argument('--data_name', default='lrssl', type=str)
    parser.add_argument('--num_neighbor', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.01)  # 0.1

    parser.add_argument('--cl_temp', type=float, default=0.1, help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--cl_dropout', type=float, default=0.1, help='Edge dropout rate for graph augmentation')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='Weight for contrastive loss in total loss')
    parser.add_argument('--lambda_cv', type=float, default=0.05, help='Weight for cross-view contrastive loss')
    parser.add_argument('--cl_mask_prob', type=float, default=0.2, help='Probability of masking node features')
    
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha parameter for Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    
    parser.add_argument('--folds', type=int, default=10, help='Number of folds to run (1-10)')

    args = parser.parse_args()
    print(args)
    
    if args.device >= 0 and not th.cuda.is_available():
        print("CUDA is not available, switching to CPU...")
        args.device = -1

    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    aucs, auprs = [], []
    for times in range(1, 2):  
        print("++++++++++++++++++times", str(times), "++++++++++++++++++++++")
        
        if not args.save_dir:
            prefix = f"neighbor_num{args.num_neighbor}"
            folder_name = args.data_name + "_" + ''.join(str(times) + 'time')
            args.save_dir = os.path.join(prefix, folder_name)
            
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        dataset = DrugDataLoader(args.data_name, args.device, symm=args.gcn_agg_norm_symm, k=args.num_neighbor)

        print("Loading dataset finished ...\n")

        start_fold = 0
        end_fold = args.folds
        if args.save_id is not None:
            start_fold = args.save_id - 1
            end_fold = args.save_id

        for cv in range(start_fold, end_fold):
            current_save_id = cv + 1
            args.save_id = current_save_id
            
            print("===============" + str(current_save_id) + "=================")
            graph_data = dataset.data_cv[cv]
            train(args, dataset, graph_data, cv)
           
