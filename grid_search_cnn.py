from cnn_split_cifar import train

if __name__ == "__main__":
    LRs = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    BSs = [64, 128, 256, 512]
    decays = [0, 1e-5, 5e-5, 1e-4]
    seed = 42
    
    for lr in LRs:
        for bs in BSs:
            for w_d in decays:
                conf = {"seed": seed, "train_bs": bs, 
                        "lr": lr, "w_decay": w_d}
                avg_acc, final_epochs, final_single_acc = train(**conf)
                print(**conf)
                print(f"avg_acc: {avg_acc}%")
                print(f"final_epochs: {final_epochs}") 
                print(f"final_single_acc: {final_single_acc}")
                print("\n\n")
