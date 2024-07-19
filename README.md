# NN-GUI-rs

Rust 练手项目，当前目标为基于项目 [**AlexiaChen/nn-rs**](https://github.com/AlexiaChen/nn-rs) 实现跨平台的 GUI 视觉数据神经网络识别、采集、封装。

## 如何开始

1. 利用 [rsproxy](https://rsproxy.cn/) 镜像安装 rust；


```bash

```

## 期望功能

- [x] 实现简单的 mnist 手写数字识别
- [ ] 实现 fusion mnist 识别
- [ ] 在 Windows 上实现基本的图片选择、分辨手写数字的 GUI
- [ ] 鼠标手写数字分辨
- [ ] 调用摄像头识别手写数字
- [ ] 采集、标注模式
- [ ] 采集结果可打包成任意格式数据集
- [ ] i18n 实现 GUI 中英文切换
- [ ] 可通过命令行、GUI 开启在线服务，并添加 WebUI
- [ ] ...

## MNIST 手写数字数据集

原格式：
http://yann.lecun.com/exdb/mnist/

csv 格式:
https://pjreddie.com/projects/mnist-in-csv/

## How to use it to recognize the handwritten digit?

```bash
./target/release/handwritten-digit-recognition ./dataset/mnist_train <path-of-image>
```

NOTE: the prefix name "2828_my_own" images are from https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/tree/master/my_own_images
the prefix name "handwrite" images are from mine created from Windows Paint

The Performance of prefix name "2828_my_own" images is better than the prefix "handwrite" images. I think that is because the digit in the the prefix name "2828_my_own" images are more bold than the digit in the prefix name "handwrite" images.

## MNIST in CSV

The format is:

```txt
label, pix-11, pix-12, pix-13, ... , pix-nn \n
newlabel, pix-11, pix-12, pix-13, ... , pix-nn \n
...
```

where pix-ij is the pixel in the i-th row and j-th column.

pix-nn is 28*28 = 784 pixel values in the range 0-255 gray values.

label is the digit represented by the image.

For the curious, this is the script to generate the csv files from the [original data](http://yann.lecun.com/exdb/mnist/)

```python
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "mnist_test.csv", 10000)
```

## FAQ

Q: Why need to seperate the train data set and test data set?

A: That is we want  to test before we train the model. Otherwise, we can let network to remember the training data set and get a high accuracy. But it is not a good model. So that it is normal case in machine learning to seperate the train data set and test data set.

## References

- https://github.com/AlexiaChen/nn-rs
- https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork
- https://makeyourownneuralnetwork.blogspot.com/
- 《Make your own neural network》 by Tariq Rashid

