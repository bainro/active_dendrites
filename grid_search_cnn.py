from cnn_split_cifar import train

if __name__ == "__main__":
    LRs = [1e-5, 1e-4, 1e-3]
    BSs = [32, 64, 128]
    decays = [0, 1e-5, 1e-4]
    seeds = [42]
    
    for lr in LRs:
        for bs in BSs:
            for w_d in decays:
                for seed in seeds:
                    conf = {"seed": seed, "train_bs": bs, 
                            "lr": lr, "w_decay": w_d}
                    running_acc, single_acc = train(**conf)
                    print(f"seed: {seed}")
                    print(f"lr: {lr}")
                    print(f"batch size: {bs}")
                    print(f"weight decay: {w_d}")
                    print(f"running avg test acc: {running_acc}") 
                    print(f"each task's individual acc: {single_acc}")
