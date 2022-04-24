import dendritic_mlp as D
from samplers import TaskRandomSampler

import torch
from torch.utils.data import DataLoader

# batch_size=256,
# val_batch_size=512,
# tasks_to_validate=[1, 4, 9, 24, 49, 99],
# loss_function=torch.nn.functional.cross_entropy,
# optimizer_class=torch.optim.Adam

num_tasks = 10
num_classes = 10
num_classes_per_task = math.floor(num_classes / num_tasks)

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
    num_segments=num_tasks
)    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = D.DendriticMLP(**conf)
    model = model.to(device)

    dataset_args = dict(
        root=os.path.expanduser("~/datasets/permutedMNIST"),
        download=True,  # Change to True if running for the first time
        seed=42,
    ),
    
    dataset = PermutedMNIST(**dataset_args)
    
    # target -> all indices for that target
    class_indices = defaultdict(list)
    for idx, (_, target) in enumerate(dataset):
        class_indices[target].append(idx)

    # task -> all indices corresponding to this task
    task_indices = defaultdict(list)
    for i in range(num_tasks):
        for j in range(num_classes_per_task):
            task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])
    
    sampler = S.TaskRandomSampler(task_indices)
    
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
