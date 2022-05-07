from sparse_split_cifar import train

if __name__ == "__main__":
    LRs = [5e-5, 1e-4, 5e-4, 1e-3]
    BSs = [64, 128, 256]
    conv_act_sparsities = [0.05, 0.1, 0.2]
    fc_act_sparsities = [0.1, 0.2]
    fc_w_sparsities = [0.5, 1]
    boosting_set = (1, 0, 0)
    seed = 42
    
    for lr in LRs:
        for bs in BSs:
            for c_a_s in conv_act_sparsities:
                for f_a_s in fc_act_sparsities:
                    for f_w_s in fc_w_sparsities:
                        conf = dict(seed=seed, train_bs=bs, 
                                    lr=lr, c_a_s=c_a_s, f_a_s=f_a_s,
                                    f_w_s=f_w_s, boost_set=boosting_set}
                        avg_acc, final_epochs, final_single_acc = train(**conf)
                        print(f"seed: {seed}")
                        print(f"lr: {lr}")
                        print(f"batch size: {bs}")
                        print(f"conv act sparsity: {c_a_s}")
                        print(f"fc act sparsity: {f_a_s}")
                        print(f"fc weight sparsity: {f_w_s}")
                        print(f"avg_acc: {avg_acc}%")
                        print(f"final_epochs: {final_epochs}") 
                        print(f"final_single_acc: {final_single_acc}")
