import numpy as np
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.datasets as datasets

import resnet_models
import project_constants

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

batch_size = 2 # 128
num_workers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_device(device)

classes = ("plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck")


def get_network():
    backbone_weights = models.ResNet152_Weights.DEFAULT

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
        cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True, 
        num_workers=num_workers, pin_memory=True, pin_memory_device="cuda" if torch.cuda.is_available() else "cpu",
    )

    cifar10_test = datasets.CIFAR10(
        root=project_constants.DATA_STORAGE_DIR, train=False, transform=resnet_models.preprocess, download=True
    )
    testloader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=2*batch_size, shuffle=False, drop_last=False, 
        num_workers=num_workers, pin_memory=True, pin_memory_device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return trainloader, testloader

def get_criterion_and_optimizer(net: torch.nn.Module, lr: float = 1e-4):

    if torch.cuda.is_available():
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    return criterion, optimizer


def train_classifier_epoch(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    loss_to_now = 0.0
    correct = 0
    total = 0
    pbar = tqdm(trainloader, total=len(trainloader), desc=f"Training...")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = net(inputs)

            total_loss = 0.0
            for i in range(outputs.shape[1]):
                loss = criterion(outputs[:, i, :], labels)
                total_loss += loss
                if i == outputs.shape[1] - 1:
                    loss_to_now += loss.item()
                    _, predicted = outputs[:, i, :].max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
        total_loss.backward()
        optimizer.step()
        
        pbar.set_postfix(
            {
                "loss": f"{loss_to_now / total:.4f}",
                "acc": f"{correct / total:.2%}",
            }
        )
    return loss_to_now, correct / total


def eval_classifier_epoch(
    net: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module
) -> tuple[float, np.ndarray]:
    loss = 0.0
    correct = torch.zeros(len(net.fc_layers), device=device)
    total = 0
    net.eval()
    pbar = tqdm(testloader, total=len(testloader), desc=f"Evaluating...")
    with torch.no_grad():
        for inputs, labels in pbar:
            
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = net(inputs)
                for i in range(outputs.shape[1]):
                    loss += criterion(outputs[:, i, :], labels).item()
                _, predicted = outputs.max(-1)
                total += labels.size(0)
                correct += (predicted == labels.view(-1, 1)).sum(dim=0)
            
            pbar.set_postfix({
                "loss": f"{loss / total:.4f}",
                "acc": f"{correct[-1].cpu().numpy() / total:.2%}"
            })
    return loss, correct.cpu().numpy() / total
                
    


def train_and_eval_classifier(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 5,
) -> None:
    net.train()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} of {n_epochs} started")

        # Training
        train_classifier_epoch(net, trainloader, criterion, optimizer)
        
        # Evaluation
        eval_loss, eval_accs = eval_classifier_epoch(net, testloader, criterion)
        print(f"Epoch {epoch + 1} of {n_epochs} finished. Loss: {eval_loss:.4f}, Accuracies:", eval_accs, sep="\n", end="\n\n")
        


    print("Finished Training")


def fgsm_attacks(net, inputs, labels, criterion, epsilon):
    net.eval()
    inputs.requires_grad = True
    # net = net.to(device="cpu")
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True) 
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs = net(inputs)
    
    perturbed_inputs = []
    for i in tqdm(range(outputs.shape[1])):
        loss = criterion(outputs[:, i, :], labels)
        loss.backward()
        with torch.no_grad():
            inputs_grad = inputs.grad
            perturbed_inputs.append(inputs + epsilon * inputs_grad.sign())
    
    return perturbed_inputs