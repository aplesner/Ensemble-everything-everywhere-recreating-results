from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.datasets as datasets

import resnet_models
import project_constants

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


backbone_weights = models.ResNet152_Weights.IMAGENET1K_V1

resnet152_ensemble = resnet_models.resnet152_ensemble(num_classes=10)

batch_size = 256
num_workers = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = resnet152_ensemble

# Load pre-trained parameters
resnet_models.set_resnet_weights(net, backbone_weights)
resnet_models.freeze_backbone(net)

# net = torch.compile(net)  # a bit faster, but does not give much
net = net.to(device)

ensemble_size = len(net.fc_layers)

cifar10_train = datasets.CIFAR10(
    root=project_constants.DATA_STORAGE_DIR, train=True, transform=resnet_models.preprocess, download=True
)
trainloader = torch.utils.data.DataLoader(
    cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
)

cifar10_test = datasets.CIFAR10(
    root=project_constants.DATA_STORAGE_DIR, train=False, transform=resnet_models.preprocess, download=True
)
testloader = torch.utils.data.DataLoader(
    cifar10_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True
)

classes = ("plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck")

criterion = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

def print_epoch(epoch: int, running_losses: list[float]) -> list[float]:
    if epoch % 20 == 19:    # print every 20 mini-batches
        print(f"[{epoch + 1}, {i + 1:5d}] losses:")
        for running_loss in running_losses:
            print(f"{running_loss / 20:.3f}", end=" ")
            # print(f"{running_loss:.3f}", end=" ")
        print()
        running_losses = [0.0] * ensemble_size
    
    return running_losses


def training_step(outputs, labels):
    losses = []
    for i in range(ensemble_size):
        loss = criterion(outputs[:, i, :], labels)
        losses.append(loss)
        running_losses[i] += loss.item()
    total_loss = sum(losses)
    total_loss.backward()
    optimizer.step()

    return losses

training_step_func = torch.compile(training_step)

for epoch in range(2):  # loop over the dataset multiple times

    running_losses = [0.0] * ensemble_size
    for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        losses = training_step(outputs=outputs, labels=labels)
        for i in range(ensemble_size):
            running_losses[i] += losses[i].item()

        # print statistics
        running_losses = print_epoch(epoch=batch_idx, running_losses=running_losses)

print("Finished Training")
