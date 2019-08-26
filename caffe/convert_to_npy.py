# -*- coding: utf-8 -*-
import numpy as np
import caffe

if __name__ == "__main__":
    data = open("imagenet_mean.binaryproto", "rb").read()

    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)

    arr = np.asarray(caffe.io.blobproto_to_array(blob))
    print (arr.shape)
    # print (type(arr)) # ndarray
    # print (arr)

    np.save("imagenet_mean.npy", arr)
