from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.datasets as datasets

import resnet_models
import project_constants

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

batch_size = 16
num_workers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ("plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck")


def get_network():
    backbone_weights = models.ResNet152_Weights.IMAGENET1K_V1

    net = resnet_models.resnet152_ensemble(num_classes=10)
    # Load pre-trained parameters
    resnet_models.set_resnet_weights(net, backbone_weights)
    resnet_models.freeze_backbone(net)

    # net = torch.compile(net, backend="eager")  # a bit faster per iteration, but has a high overhead
    return net.to(device)



def get_data_loaders():
    cifar10_train = datasets.CIFAR10(
        root=project_constants.DATA_STORAGE_DIR, train=True, transform=resnet_models.preprocess, download=True
    )
    trainloader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    cifar10_test = datasets.CIFAR10(
        root=project_constants.DATA_STORAGE_DIR, train=False, transform=resnet_models.preprocess, download=True
    )
    testloader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=2*batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    return trainloader, testloader

def get_criterion_and_optimizer(net: torch.nn.Module, lr: float = 1e-4):
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer



def print_epoch(epoch: int, running_losses: torch.Tensor) -> None:
    print(f"[{epoch + 1}] losses:")
    for running_loss in running_losses:
        print(f"{running_loss / 20:.3f}", end=" ")
        # print(f"{running_loss:.3f}", end=" ")
    print()
    running_losses.zero_()
    

def train_classifier(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> None:

    n_epochs = 2
    ensemble_size = len(net.fc_layers)
    print_progress_every = len(trainloader) // 10
    net.train()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch + 1} of {n_epochs} started")

        running_losses = torch.zeros(ensemble_size, dtype=torch.float32, device=device, requires_grad=False)
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            # training step
            total_loss = 0.0
            for i in range(ensemble_size):
                loss = criterion(outputs[:, i, :], labels)
                total_loss += loss
                running_losses[i] += loss.item()
            total_loss.backward()
            optimizer.step()
            
            # print statistics
            if batch_idx % print_progress_every == print_progress_every - 1:
                print_epoch(epoch=batch_idx, running_losses=running_losses)

    print("Finished Training")


