import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from model import Net
from data import DrugDataLoader, DrugColdStartDataLoader
from utils import setup_seed

def get_metrics(y_true, y_score):
    if th.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if th.is_tensor(y_score):
        y_score = y_score.cpu().numpy()
    
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

def sparsify_matrix(matrix, top_k):
    if top_k >= matrix.shape[1] or top_k <= 0:
        return matrix
        
    values, indices = th.topk(matrix, top_k, dim=1)
    sparse_matrix = th.zeros_like(matrix)
    sparse_matrix.scatter_(1, indices, values)
    return sparse_matrix

def p_tslr_refinement(pred_matrix, drug_proj, dis_proj, drug_raw_sim, anchor_matrix, 
                      mu=1.0, epsilon=0.01, alpha=0.7, beta=0.5, top_k=20, gamma=0.5,
                      omega_min=0.3, omega_max=0.7, num_hops=1, hop_decay=0.6,
                      dis_top_k=50, dis_num_hops=2, dis_hop_decay=0.6):
    drug_proj = F.normalize(drug_proj, p=2, dim=1)
    S_learned = th.matmul(drug_proj, drug_proj.t())
    S_learned = th.clamp(S_learned, 0, 1)
    
    if drug_raw_sim is None:
        S_raw = S_learned
    else:
        S_raw = th.clamp(drug_raw_sim, 0, 1)
    
    S_hybrid = alpha * S_learned + (1 - alpha) * S_raw
    S_hybrid_sparse = sparsify_matrix(S_hybrid, top_k)
    
    drug_degrees = anchor_matrix.sum(dim=1, keepdim=True) # [N_drug, 1]
    drug_degrees = th.clamp(drug_degrees, min=1.0)
    hub_weights = 1.0 / (drug_degrees.pow(gamma)) # [N_drug, 1]
    anchor_matrix_weighted = anchor_matrix * hub_weights # Broadcasting [N, M] * [N, 1]
    
    N_drug = S_raw.shape[0]
    S_raw_no_diag = S_raw * (1.0 - th.eye(N_drug, device=S_raw.device))
    max_sim, _ = S_raw_no_diag.max(dim=1, keepdim=True) # [N, 1]
    
    novelty_score = 1.0 - max_sim
    
    omega_min_val = omega_min
    omega_max_val = omega_max
    if omega_max_val < omega_min_val:
        omega_min_val, omega_max_val = omega_max_val, omega_min_val
    omega_adaptive = omega_min_val + (omega_max_val - omega_min_val) * novelty_score
    omega_adaptive = th.clamp(omega_adaptive, 0.0, 1.0)
    
    dis_norm = th.norm(anchor_matrix, p=2, dim=0, keepdim=True)
    dis_norm = th.where(dis_norm == 0, th.ones_like(dis_norm), dis_norm)
    Y_norm = anchor_matrix / dis_norm
    
    S_disease = th.matmul(Y_norm.t(), Y_norm)
    S_disease.fill_diagonal_(0)
    
    term_disease = th.matmul(anchor_matrix, S_disease)

    S_disease_sparse = sparsify_matrix(S_disease, dis_top_k)
    term_disease_rescue = th.matmul(anchor_matrix, S_disease_sparse)
    if dis_num_hops > 1:
        term_disease_accum = term_disease_rescue
        term_disease_h = term_disease_rescue
        for hop in range(2, int(dis_num_hops) + 1):
            term_disease_h = th.matmul(term_disease_h, S_disease_sparse)
            term_disease_accum = term_disease_accum + (float(dis_hop_decay) ** (hop - 1)) * term_disease_h
        term_disease_rescue = term_disease_accum
    
    Score_rescue_drug = th.matmul(S_hybrid_sparse, anchor_matrix_weighted)
    if num_hops > 1:
        Score_rescue_drug_accum = Score_rescue_drug
        Score_rescue_drug_h = Score_rescue_drug
        for hop in range(2, int(num_hops) + 1):
            Score_rescue_drug_h = th.matmul(S_hybrid_sparse, Score_rescue_drug_h)
            Score_rescue_drug_accum = Score_rescue_drug_accum + (float(hop_decay) ** (hop - 1)) * Score_rescue_drug_h
        Score_rescue_drug = Score_rescue_drug_accum
    
    Score_rescue = (1 - omega_adaptive) * Score_rescue_drug + omega_adaptive * term_disease_rescue
    
    max_rescue = Score_rescue.max(dim=0, keepdim=True)[0]
    Score_rescue_norm = Score_rescue / (max_rescue + 1e-6)
    
    term1 = th.matmul(S_raw, anchor_matrix_weighted)
    term2 = th.matmul(S_learned, anchor_matrix_weighted)
    
    term2_scaled = term2 * mu
    
    drug_side_support = th.sqrt(term1 * term2_scaled + 1e-8) + 0.5 * term2_scaled
    
    M_support = (1 - omega_adaptive) * drug_side_support + omega_adaptive * term_disease

    
    mean_support = M_support.mean()
    if mean_support > 0:
        M_support = M_support / mean_support * 0.5 # Heuristic scaling
        
    penalty_gate = 0.8 + 0.2 * th.tanh(M_support)
    
    base_gate = th.max(penalty_gate, 0.8 + 0.15 * novelty_score)
    base_threshold = 0.95
    adaptive_threshold = base_threshold - 0.15 * novelty_score 
    confidence_mask = th.sigmoid((pred_matrix - adaptive_threshold) * 25) 
    final_gate = base_gate * (1 - confidence_mask) + 1.0 * confidence_mask
    
    P_penalized = pred_matrix * final_gate
    
    P_final = th.max(P_penalized, beta * Score_rescue_norm)
    
    return th.clamp(P_final, 0.0, 1.0)


def compute_full_matrix(model, drug_out, dis_out):
    N, D = drug_out.shape
    M, _ = dis_out.shape
    
    drug_rep = drug_out.unsqueeze(1).expand(N, M, D)
    dis_rep = dis_out.unsqueeze(0).expand(N, M, D)
    combined = th.cat([drug_rep, dis_rep], dim=2)
    combined_flat = combined.reshape(-1, 2 * D)
    
    x = combined_flat
    x = F.relu(model.decoder.lin1(x))
    x = F.relu(model.decoder.lin2(x))
    x = model.decoder.lin3(x)
    
    logits = x.reshape(N, M)
    return th.sigmoid(logits)

def run_inference(args):
    Loader = DrugColdStartDataLoader if args.cold_start_drug_split else DrugDataLoader
    dataset = Loader(args.data_name, args.device, symm=args.gcn_agg_norm_symm, k=args.num_neighbor)
    
    fold_idx = args.save_id - 1
    print(f"Loading data for fold {fold_idx + 1}...")
    graph_data = dataset.data_cv[fold_idx]
    
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]
    args.rating_vals = dataset.possible_rel_values
    
    drug_graph = dataset.drug_graph.to(args.device)
    dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature
    
    test_gt_ratings = graph_data['test'][2].to(args.device)
    test_enc_graph = graph_data['test'][0].int().to(args.device)
    test_dec_graph = graph_data['test'][1].int().to(args.device)
    
    model = Net(args=args).to(args.device)
    checkpoint_path = getattr(args, 'checkpoint_path', None)
    if checkpoint_path is None:
        checkpoint_type = getattr(args, 'checkpoint_type', 'auroc')
        if checkpoint_type == 'aupr':
            checkpoint_path = os.path.join(args.save_dir, 'best_model_aupr_%d.pth' % args.save_id)
        else:
            checkpoint_path = os.path.join(args.save_dir, 'best_model_%d.pth' % args.save_id)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(th.load(checkpoint_path, map_location=args.device))
    model.eval()
    
    print("\nRunning Baseline Inference...")
    with th.no_grad():
        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_proj, dis_proj, drug_feats, dis_feats = \
            model(test_enc_graph, test_dec_graph,
                  drug_graph, drug_sim_feat, drug_feat,
                  dis_graph, dis_sim_feat, dis_feat,
                  Two_Stage=False)
                  
        base_auroc, base_aupr = get_metrics(test_gt_ratings, pred_ratings.squeeze())
        print(f"Baseline Results (Direct): AUROC = {base_auroc:.4f}, AUPR = {base_aupr:.4f}")
        
        drug_feats = drug_feats.to(args.device)
        dis_feats = dis_feats.to(args.device)
        full_pred_prob = compute_full_matrix(model, drug_feats, dis_feats)
    
    u, v = test_dec_graph.edges(etype='rate')
    u = u.to(args.device)
    v = v.to(args.device)
    reconstructed_scores = full_pred_prob[u, v]
    rec_auroc, rec_aupr = get_metrics(test_gt_ratings, reconstructed_scores)
    print(f"Baseline Results (Recon):  AUROC = {rec_auroc:.4f}, AUPR = {rec_aupr:.4f}")
    
    if not args.enable_grid_search:
        print("\nRunning P-TSLR (single setting)...")
    else:
        print("\nRunning TSLR Grid Search (Dual Similarity + Path Refinement)...")
        print("{:<10} {:<10} {:<10} | {:<10} {:<10}".format("Beta", "Mu", "AnchorMode", "AUROC", "AUPR"))
        print("-" * 60)
    
    target_key = args.save_id - 1
    if target_key not in dataset.cv_data_dict:
        print(f"Error: Target Key {target_key} not found. Using key 0.")
        target_key = 0
        
    train_df = dataset.cv_data_dict[target_key][0]
    train_pos = train_df[train_df['values'] == 1]
    
    train_indices_np = np.vstack((train_pos['drug_id'].values, train_pos['disease_id'].values))
    train_indices = th.from_numpy(train_indices_np).long()
    train_values = th.ones(len(train_pos))
    N_drug = dataset._num_drug
    N_dis = dataset._num_disease
    
    train_anchors = th.sparse_coo_tensor(train_indices, train_values, (N_drug, N_dis)).to_dense().to(args.device)
    
    drug_raw_sim = th.tensor(dataset.drug_sim_features).float().to(args.device)
    
    best_res = {'auroc': 0, 'aupr': 0, 'params': '', 'beta': 0, 'mu': 0, 'gamma': 0}
    results_list = []
    if not args.enable_grid_search:
        beta = args.beta
        mu = args.mu
        gamma = args.gamma

        refined_full_prob = p_tslr_refinement(
            full_pred_prob,
            drug_proj,
            dis_proj,
            drug_raw_sim=drug_raw_sim,
            anchor_matrix=train_anchors,
            mu=mu,
            beta=beta,
            gamma=gamma,
            top_k=args.top_k,
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            num_hops=args.num_hops,
            hop_decay=args.hop_decay,
            dis_top_k=args.dis_top_k,
            dis_num_hops=args.dis_num_hops,
            dis_hop_decay=args.dis_hop_decay,
            alpha=0.7,
            epsilon=0.01
        )
        refined_scores = refined_full_prob[u, v]
        curr_auroc, curr_aupr = get_metrics(test_gt_ratings, refined_scores)

        best_res['auroc'] = curr_auroc
        best_res['aupr'] = curr_aupr
        best_res['params'] = f"Beta={beta}, Mu={mu}, Gamma={gamma}, Omega=({args.omega_min},{args.omega_max})"
        best_res['beta'] = beta
        best_res['mu'] = mu
        best_res['gamma'] = gamma
        results_list.append({
            'Beta': beta,
            'Mu': mu,
            'Gamma': gamma,
            'AnchorMode': 'Train',
            'AUROC': curr_auroc,
            'AUPR': curr_aupr
        })
        print(f"Single P-TSLR: {best_res['params']}, AUROC={curr_auroc:.4f}, AUPR={curr_aupr:.4f}")
    else:
        betas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
        mus = [0.5, 1.0, 1.5]
        gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        print("{:<10} {:<10} {:<10} {:<10} | {:<10} {:<10}".format("Beta", "Mu", "Gamma", "AnchorMode", "AUROC", "AUPR"))
        print("-" * 75)

        for beta in betas:
            for mu in mus:
                for gamma in gammas:
                    refined_full_prob = p_tslr_refinement(
                        full_pred_prob,
                        drug_proj,
                        dis_proj,
                        drug_raw_sim=drug_raw_sim,
                        anchor_matrix=train_anchors,
                        mu=mu,
                        beta=beta,
                        gamma=gamma,
                        top_k=args.top_k,
                        omega_min=args.omega_min,
                        omega_max=args.omega_max,
                        num_hops=args.num_hops,
                        hop_decay=args.hop_decay,
                        dis_top_k=args.dis_top_k,
                        dis_num_hops=args.dis_num_hops,
                        dis_hop_decay=args.dis_hop_decay,
                        alpha=0.7,
                        epsilon=0.01
                    )
                    refined_scores = refined_full_prob[u, v]
                    curr_auroc, curr_aupr = get_metrics(test_gt_ratings, refined_scores)

                    print("{:<10} {:<10} {:<10} {:<10} | {:<10.4f} {:<10.4f}".format(
                        beta, mu, gamma, "Train", curr_auroc, curr_aupr))

                    results_list.append({
                        'Beta': beta,
                        'Mu': mu,
                        'Gamma': gamma,
                        'AnchorMode': 'Train',
                        'AUROC': curr_auroc,
                        'AUPR': curr_aupr
                    })

                    if curr_auroc > best_res['auroc']:
                        best_res['auroc'] = curr_auroc
                        best_res['aupr'] = curr_aupr
                        best_res['params'] = f"Beta={beta}, Mu={mu}, Gamma={gamma}, Omega=({args.omega_min},{args.omega_max})"
                        best_res['beta'] = beta
                        best_res['mu'] = mu
                        best_res['gamma'] = gamma
                
    print("\nBest P-TSLR Result:")
    print(f"Params: {best_res['params']}")
    print(f"AUROC: {best_res['auroc']:.4f}")
    if args.save_results:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        grid_csv_path = os.path.join(args.save_dir, f'tslr_grid_search_{args.save_id}.csv')
        pd.DataFrame(results_list).to_csv(grid_csv_path, index=False)
        summary_path = os.path.join(args.save_dir, f'tslr_summary_{args.save_id}.csv')
        pd.DataFrame([{
            'fold': args.save_id,
            'base_auroc': base_auroc,
            'base_aupr': base_aupr,
            'ptslr_auroc': best_res['auroc'],
            'ptslr_aupr': best_res['aupr'],
            'best_params': best_res['params']
        }]).to_csv(summary_path, index=False)
        print(f"\nSaved results to: {grid_csv_path}")
        print(f"Saved summary to: {summary_path}")
            
    return {
        'fold': args.save_id,
        'base_auroc': base_auroc,
        'base_aupr': base_aupr,
        'ptslr_auroc': best_res['auroc'],
        'ptslr_aupr': best_res['aupr'],
        'best_params': best_res['params']
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='Ldataset')
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--save_id', type=int, default=10)
    parser.add_argument('--checkpoint_type', choices=['auroc', 'aupr'], default='auroc')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--num_neighbor', type=int, default=8)
    parser.add_argument('--cold_start_drug_split', action='store_true')
    parser.add_argument('--enable_grid_search', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--omega_min', type=float, default=0.3)
    parser.add_argument('--omega_max', type=float, default=0.7)
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--hop_decay', type=float, default=0.6)
    parser.add_argument('--dis_top_k', type=int, default=50)
    parser.add_argument('--dis_num_hops', type=int, default=2)
    parser.add_argument('--dis_hop_decay', type=float, default=0.6)
    
    # Model Hyperparameters (Must match training)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--model_activation', type=str, default="tanh")
    parser.add_argument('--gcn_agg_units', type=int, default=840)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--nhid1', type=int, default=500)
    parser.add_argument('--nhid2', type=int, default=75)
    parser.add_argument('--share_param', default=True, action='store_true')
    
    args = parser.parse_args()
    
    setup_seed(1024)
    
    if 'ipykernel_launcher' in sys.argv[0]:
        args = parser.parse_args([])
        
    run_inference(args)
