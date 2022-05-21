import utils
from engine import train_one_epoch
from dataset import ShapeDataset
import torch
import torch.utils.data

num_classes = 4  # 0 for background

model = utils.get_instance_segmentation_model(num_classes).double()
model.load_state_dict(torch.load("./pretrained_model/intro2cv_maskrcnn_pretrained.pth", map_location='cpu'))

dataset = ShapeDataset(10)

torch.manual_seed(233)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=utils.collate_fn)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 3
device = torch.device('cpu')

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    torch.save(model.state_dict(), "results/maskrcnn_" + str(epoch) + ".pth")
