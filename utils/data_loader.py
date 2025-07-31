from inputs import mnist_input, fashion_mnist_input, cifar10_input

def get_train(dataset, model='fcnn'):
    if dataset == 'mnist':
        return mnist_input.get_train_data(model)
    elif dataset == 'fashion_mnist':
        return fashion_mnist_input.get_train_data(model)
    elif dataset == 'cifar10':
        return cifar10_input.get_train_data(model)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_test(dataset, model='fcnn'):
    if dataset == 'mnist':
        return mnist_input.get_test_data(model)
    elif dataset == 'fashion_mnist':
        return fashion_mnist_input.get_test_data(model)
    elif dataset == 'cifar10':
        return cifar10_input.get_test_data(model)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")