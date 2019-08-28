from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import caffe
import torch
import hickle as hkl


def create_feature_dataset(net, datalist, dbprefix, mean_values, device):
    with open(datalist) as fr:
        lines = fr.readlines()
    lines = [line.rstrip() for line in lines]
    feats = []
    labels = []
    for line_i, line in enumerate(lines):
        img_path, label = line.split()
        img = caffe.io.load_image(img_path)
        img = np.transpose(img, (2, 0, 1))
        # to tensor
        img = torch.from_numpy(img)
        # mean
        normalize = transforms.Normalize(mean=(mean_values[0], mean_values[1], mean_values[2]), std=(0.5, 0.5, 0.5))
        img = normalize(img)
        img = img * 255.0
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        net = net.eval()

        feat = net(img)

        feats.append(feat)
        label = int(label)
        labels.append(label)
        if (line_i + 1) % 100 == 0:
            print("processed", line_i + 1)
    feats = np.asarray(feats)
    labels = np.asarray(labels)
    hkl.dump(feats, './features/' + dbprefix + "_features.hkl", mode="w")
    hkl.dump(labels, './features/' + dbprefix + "_labels.hkl", mode="w")


def main():
    torch.cuda.empty_cache()

    num_classes = 2
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_ft = models.resnet18(pretrained=True)
    num_feature = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_feature, num_classes)
    model_ft = model_ft.to(device)
    create_feature_dataset(model_ft, datalist='../dataset/dogs-vs-cats/train.txt', dbprefix='cac_',
                           mean_values=[103.939, 116.779, 123.68], device=device)


if __name__ == '__main__':
    main()






