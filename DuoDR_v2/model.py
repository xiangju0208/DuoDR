from layers import *

th.set_printoptions(profile="full")


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.TGCN = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                   args.src_in_units,  # 909
                                   args.dst_in_units,  # 909
                                   args.gcn_agg_units,  # 1800
                                   args.gcn_out_units,  # 75
                                   args.dropout,  # 0.3
                                   args.gcn_agg_accum,  # sum
                                   agg_act=self._act,  # Tanh()
                                   share_user_item_param=args.share_param,  # True
                                   device=args.device))
        self.gcn_agg_accum = args.gcn_agg_accum  # sum
        self.rating_vals = args.rating_vals  # sum[0, 1]
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units  # 1800
        self.src_in_units = args.src_in_units  # 909
        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                       args.gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       ini=False,
                                       device=args.device))

        self.FGCN = FGCN(args.fdim_drug,
                         args.fdim_disease,
                         args.nhid1,
                         args.nhid2,
                         args.dropout)

        self.attention = Attention(args.gcn_out_units)
        self.gatedfusion = GatedMultimodalLayer(args.gcn_out_units, args.gcn_out_units, args.gcn_out_units)
        self.decoder = MLPDecoder(in_units=args.gcn_out_units)
        
        self.projection_head = nn.Sequential(
            nn.Linear(args.gcn_out_units, args.gcn_out_units),
            nn.ReLU(),
            nn.Linear(args.gcn_out_units, args.gcn_out_units)
        )
        
        self.rating_vals = args.rating_vals

    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                Two_Stage=False):

        drug_out, dis_out = None, None
        for i in range(0, self.layers):
            drug_o, dis_o = self.TGCN[i](enc_graph, drug_feat, dis_feat, Two_Stage)
            if i == 0:
                drug_out = drug_o
                dis_out = dis_o

            else:
                drug_out += drug_o / float(i + 1)
                dis_out += dis_o / float(i + 1)

            drug_feat = drug_o
            dis_feat = dis_o

        drug_sim_out, dis_sim_out = self.FGCN(drug_graph, drug_sim_feat,
                                              dis_graph, disease_sim_feat)
        
        drug_feats = th.stack([drug_out, drug_sim_out], dim=1)
        drug_feats, att_drug = self.attention(drug_feats)

        dis_feats = th.stack([dis_out, dis_sim_out], dim=1)
        dis_feats, att_dis = self.attention(dis_feats)

        pred_ratings = self.decoder(dec_graph, drug_feats, dis_feats)
        
        drug_proj = self.projection_head(drug_feats)
        dis_proj = self.projection_head(dis_feats)
        
        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_proj, dis_proj, drug_feats, dis_feats
