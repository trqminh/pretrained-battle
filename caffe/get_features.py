import feature_extract as fe


def run_alexnet():
    alexnet = fe.CaffeFeatureExtractor(
            model_path="./pretrained_models/alexnet_deploy.prototxt",
            pretrained_path="./pretrained_models/alexnet.caffemodel",
            blob="fc6",
            crop_size=227,
            meanfile_path="imagenet_mean.npy"
            )
    fe.create_dataset(net=alexnet, datalist="../dataset/dogs-vs-cats/train.txt", dbprefix="alexnet_train")
    fe.create_dataset(net=alexnet, datalist="../dataset/dogs-vs-cats/test.txt", dbprefix="alexnet_test")


def run_vgg16_fc7():
    vgg16 = fe.CaffeFeatureExtractor(
            model_path="./pretrained_models/vgg16_deploy.prototxt",
            pretrained_path="./pretrained_models/vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    fe.create_dataset(net=vgg16, datalist="../dataset/dogs-vs-cats/train.txt", dbprefix="vgg16_fc7_train")
    fe.create_dataset(net=vgg16, datalist="../dataset/dogs-vs-cats/test.txt", dbprefix="vgg16_fc7_test")


def run_vgg16_fc6():
    vgg16 = fe.CaffeFeatureExtractor(
            model_path="./pretrained_models/vgg16_deploy.prototxt",
            pretrained_path="./pretrained_models/vgg16.caffemodel",
            blob="fc6",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    fe.create_dataset(net=vgg16, datalist="../dataset/dogs-vs-cats/train.txt", dbprefix="vgg16_fc6_train")
    fe.create_dataset(net=vgg16, datalist="../dataset/dogs-vs-cats/test.txt", dbprefix="vgg16_fc6_test")


def run_googlenet():
    googlenet = fe.CaffeFeatureExtractor(
            model_path="./pretrained_models/googlenet_deploy.prototxt",
            pretrained_path="./pretrained_models/googlenet.caffemodel",
            blob="pool5/7x7_s1",
            crop_size=224,
            mean_values=[104.0, 117.0, 123.0]
            )
    fe.create_dataset(net=googlenet, datalist="../dataset/dogs-vs-cats/train.txt", dbprefix="googlenet_train")
    fe.create_dataset(net=googlenet, datalist="../dataset/dogs-vs-cats/test.txt", dbprefix="googlenet_test")


if __name__ == "__main__":
    run_alexnet()
    run_vgg16_fc7()
    run_vgg16_fc6()
    run_googlenet()
