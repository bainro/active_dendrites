import dendritic_mlp as D

import torch
from torch.utils.data import DataLoader

# batch_size=256,
# val_batch_size=512,
# tasks_to_validate=[1, 4, 9, 24, 49, 99],
# seed=42,
# loss_function=torch.nn.functional.cross_entropy,
# optimizer_class=torch.optim.Adam

conf = dict(
    input_size=784,
    output_size=10,  # Single output head shared by all tasks
    hidden_sizes=[2048, 2048],
    dim_context=784,
    kw=True,
    kw_percent_on=0.05,
    dendrite_weight_sparsity=0.0,
    weight_sparsity=0.5,
    context_percent_on=0.1,
    num_segments=10
)    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = D.DendriticMLP(**conf)
    model = model.to(device)

    if dataset is None:
        dataset = cls.load_dataset(config, train=True)

    sampler = cls.create_train_sampler(config, dataset)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=sampler is None,
        num_workers=config.get("workers", 0),
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=config.get("train_loader_drop_last", True),
    )

    # import pdb; pdb.set_trace()
    print("SCRIPT FINISHED!")
