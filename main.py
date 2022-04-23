    import .dendritic_mlp
    
    ### base experiment
    
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[64, 64],
        num_segments=NUM_TASKS,
        dim_context=1024,  # Note: with the Gaussian dataset, `dim_context` was
        # 2048, but this shouldn't effect results
        kw=True,
        # dendrite_sparsity=0.0,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=1,
    tasks_to_validate=(0, 1, 2),  # Tasks on which to run validate
    num_tasks=NUM_TASKS,
    num_classes=10 * NUM_TASKS,
    distributed=False,
    seed=42,

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=0.001),
    
    
    #### prototype experiment
    
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
