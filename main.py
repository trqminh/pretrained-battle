import hickle as hkl
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import cv2
import caffe


def run_caffe(model_name):
    print("==> loading train data from %s" % ("./caffe/features/" + model_name + "_train_(features|labels).hkl"))
    train_features = hkl.load("./caffe/features/" + model_name + "_train_features.hkl")
    train_labels = hkl.load("./caffe/features/" + model_name + "_train_labels.hkl")
    train_sample = train_features.shape[0]
    train_features = train_features.reshape(train_sample, -1)
    print("train_features.shape =", train_features.shape)
    print("train_labels.shape =", train_labels.shape)

    svm = LinearSVC(C=1.0)

    print("==> cross validation")
    scores = cross_val_score(svm, train_features, train_labels, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    svm.fit(train_features, train_labels)

    print("==> loading test data from %s" % ("./caffe/features/" + model_name + "_test_(features|labels).hkl"))
    test_features = hkl.load("./caffe/features/" + model_name + "_test_features.hkl")
    test_features = test_features.reshape(test_features.shape[0], -1)
    print("==> predicting and writing")
    predicted_labels = svm.predict(test_features)

    with open("./dataset/dogs-vs-cats/test.txt") as fr:
        lines = fr.readlines()
    image_ids = []
    labels = []
    for line in lines:
        image_path = line.split()[0]
        image_name = line.split("/")[-1]
        image_id = image_name.split(".")[0]
        image_id = int(image_id)
        label = int(image_name[-2])
        labels.append(label)
        image_ids.append(image_id)

    assert len(image_ids) == len(predicted_labels)
    with open('./caffe/predictions/' + model_name + "_predict.txt", "w") as fw:
        fw.write("id,label\n")
        for i in range(len(image_ids)):
            fw.write("%d,%d\n" % (image_ids[i], predicted_labels[i]))

    # labels = np.array(labels)
    # test_acc = (predicted_labels == labels).sum() / labels.shape[0]
    # print('Test acc: ', test_acc)


def run_pytorch(model_name):
    pass


if __name__ == '__main__':
    # print('****** Alexnet ******')
    # run_caffe("alexnet")
    # print('****** VGG16 FC7 ******')
    # run_caffe("vgg16_fc7")
    # print('****** VGG16 FC6 ******')
    # run_caffe("vgg16_fc6")
    # print('****** GoogLenet ******')
    # run_caffe("googlenet")
    img = cv2.imread('/home/aioz-interns/Downloads/data/dogs-vs-cats/train/cat.3047.jpg')
    print(img.shape)
    img = caffe.io.load_image('~/Downloads/data/dogs-vs-cats/train/cat.3046.jpg')
    print(img.shape)



