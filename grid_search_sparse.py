from sparse_split_cifar import train

if __name__ == "__main__":
    LRs = [1e-5, 1e-4, 1e-3]
    BSs = [32, 64, 128]
    conv_act_sparsities = [0.2]#, 0.1]
    fc_act_sparsities = [0.1, 0.2]
    fc_w_sparsities = [1, 0.5]
    k_inference_factors = [0.75, 1.125, 1.5]
    boost_strength = [0, 0.75, 1.5]
    boost_strength_factor = [0, 0.4, 0.85]
    boosting_pms = zip(k_inference_factors, boost_strength, boost_strength_factor)
    seed = 42
    
    for lr in LRs:
        for bs in BSs:
            for c_a_s in conv_act_sparsities:
                for f_a_s in fc_act_sparsities:
                    for f_w_s in fc_w_sparsities:
                        for boosting_set in boosting_pms:
                            (k_i_f, b_str, b_str_f) = boosting_set
                            conf = {"seed": seed, "train_bs": bs, 
                                    "lr": lr, "c_a_s": c_a_s, "f_a_s": f_a_s,
                                    "f_w_s": f_w_s, "boost_set": boosting_set}
                            avg_acc, final_epochs, final_single_acc = train(**conf)
                            print(f"seed: {seed}")
                            print(f"lr: {lr}")
                            print(f"batch size: {bs}")
                            print(f"conv act sparsity: {c_a_s}")
                            print(f"fc act sparsity: {f_a_s}")
                            print(f"fc weight sparsity: {f_w_s}")
                            print(f"k inference factor: {k_i_f}")
                            print(f"boosting strength: {b_str}")
                            print(f"boosting strength factor: {b_str_f}")
                            print(f"avg_acc: {avg_acc}%")
                            print(f"final_epochs: {final_epochs}") 
                            print(f"final_single_acc: {final_single_acc}")
