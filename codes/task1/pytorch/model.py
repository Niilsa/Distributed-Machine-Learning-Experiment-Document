import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from MyOptimizer import GDOptimizer, SGDOptimizer, AdamOptimizer, seed_everything
import plotly.express as px
import pandas as pd
import plotly.io as pio

def plot_losses(losses, filename):
    # Create a sequence of numbers as x-axis
    x = list(range(1, len(losses) + 1))

    # Multiply each value in x-axis array by 20
    x_multiplied = [val * 20 for val in x]

    # Create a DataFrame with multiplied x-axis values and losses
    data = pd.DataFrame({'X': x_multiplied, 'Loss': losses})

    # Create a line plot using Plotly Express
    fig = px.line(data, x='X', y='Loss')

    # Customize the plot layout
    fig.update_layout(
        title=filename,
        xaxis_title='Iteration',
        yaxis_title='Loss',
        template='plotly_white',
        width=800
    )

    # Save the figure as an image file (e.g., PNG)
    #fig.write_image(f'report_images/{filename}.png')
    
    pio.write_image(fig, f'report_images/{filename}.png', width=800, height=400, scale=2)

    # Show the plot
    #fig.show()

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28)
        """
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        # flatten the feature map
        out = out.flatten(1)
        # fc layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

def train(model, dataloader, optimizer, loss_fn, num_epochs=1):
    print("Start training ...")
    loss_total = 0.
    losses = []
    model.train()
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            # with dist_autograd.context() as context_id:
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            loss_total += loss.item()
            if i % 20 == 19:    
                print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))
                losses.append(loss_total / 20)
                loss_total = 0.0
    
    print("Training Finished!")
    return losses

def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def main():
    seed_everything(42)

    model = Net(in_channels=1, num_classes=10)
    model.cuda()

    DATA_PATH = "./data"

    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
                )

    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()  #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #optimizer = GDOptimizer(model.parameters(), lr=0.01)
    optimizer = SGDOptimizer(model.parameters(), lr=0.01)
    #optimizer = AdamOptimizer(model.parameters(), lr=0.01, beta1=0.8, beta2=0.999)

    losses = train(model, train_loader, optimizer, loss_fn)
    test(model, test_loader)
    #plot_losses(losses, 'ADAM loss')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()

## ADAM - accuracy = 96.5, lr=0.01, beta1=0.8, beta2=0.999
## GD - 98.44%
## SGD - 91.36%