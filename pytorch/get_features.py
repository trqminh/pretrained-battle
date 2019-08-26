from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn


def create_feature_dataset(net, datalist, dbprefix):
    with open(datalist) as fr:
        lines = fr.readlines()
    lines = [line.rstrip() for line in lines]
    feats = []
    labels = []
    for line_i, line in enumerate(lines):
        img_path, label = line.split()
        img = caffe.io.load_image(img_path)
        feat = net.extract_feature(img)
        feats.append(feat)
        label = int(label)
        labels.append(label)
        if (line_i + 1) % 100 == 0:
            print("processed", line_i + 1)
    feats = np.asarray(feats)
    labels = np.asarray(labels)
    hkl.dump(feats, './features/' + dbprefix + "_features.hkl", mode="w")
    hkl.dump(labels, './features/' + dbprefix + "_labels.hkl", mode="w")


if __name__ == '__main__':
    num_classes = 2

    img_dataset = datasets.ImageFolder('~/Downloads/data/dogs-vs-cats/train/', transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(103.939, 116.779, 123.68))
    ]))
    data_loader = DataLoader(dataset=img_dataset, batch_size=1, shuffle=True, num_workers=4)

    model_ft = models.alexnet(pretrained=True)
    num_feature = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_feature, num_classes)




