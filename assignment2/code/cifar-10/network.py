import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        # >>>---------------- define a network ---------------->>>
        self.model = nn.Sequential(
            # PART 1
            nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # PART 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # PART 3
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=1600, out_features=512), nn.ReLU(),
            nn.Linear(in_features=512,  out_features=10),
        )
        # <<<---------------- define a network ----------------<<<

    def forward(self, x):
        # >>> network forwarding >>>
        x = self.model(x)
        # <<< network forwarding <<<
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
