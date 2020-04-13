from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN

from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152

import torchtext

import torch
import torch.utils.data
import utils

from myTransforms import get_transform
from myDataset import IcartoonDataset

def collate_fn(batch):
    return tuple(zip(*batch))



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    i = 1
    for img, box in train_loader:
        imgs = list(image.to(device) for image in img)
        boxs = [{k: v.to(device) for k, v in t.items()} for t in box]
        i += 1
        loss_dict = model(imgs,boxs)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        print(loss_value)
        print('-'*50)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # print(loss)
        if i > 10:
            return
    # model()

def main():
    dataset = IcartoonDataset(transforms=True)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    test_loader = torch.utils.data.DataLoader(dataset_test,num_workers=4,collate_fn=collate_fn)

    num_classes = 1
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

    # 构造一个优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # 和学习率调度程序
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    epoch = 5
    for e in range(epoch):
        train(model, device='cpu', train_loader=test_loader, optimizer=optimizer, epoch=e)
        lr_scheduler.step()
        torch.save(model.state_dict(), './{}.model'.format(e))

    pass


if __name__ == '__main__':
    main()



