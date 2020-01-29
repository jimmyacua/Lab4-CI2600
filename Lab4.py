import torch.nn as nn
import torch
import torch.optim as optim
import struct as st
from PIL import Image
import os
import random
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PIL import Image


def read_idx(archivo):
    data = open(archivo, 'rb')

    # magic_number = int.from_bytes(data.read(4), byteorder="big", signed=True)
    data.seek(0)
    magic = st.unpack('>4B', data.read(4))
    # print(magic[3])

    if magic[3] == 3:
        images = int.from_bytes(data.read(4), byteorder="big", signed=True)

        rows = int.from_bytes(data.read(4), byteorder="big", signed=True)

        columns = int.from_bytes(data.read(4), byteorder="big", signed=True)

        '''print('Magic number: {}\nNumber of images: {}\nRows: {}\nColumns: {}'.format(
            magic,
            images,
            rows,
            columns
        ))'''

        binary_vector = data.read(images * rows * columns)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        tensor = tensor.view(images, rows, columns)
        # print(tensor)
        return tensor

    elif magic[3] == 1:
        labels = int.from_bytes(data.read(4), byteorder="big", signed=True)

        '''print('Magic number: {}\nNumber of labels: {}'.format(
            magic,
            labels,
        ))'''

        binary_vector = data.read(labels)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        # print("LABELS", tensor.view(labels))

        return tensor


def save_images(images):
    una = Image.new('L', (28, 28))
    for i in range(0, 5):
        una.putdata(list(images[i].view(-1)))  # el -1 convierte a una dimension
        una.show()
        una.save(str(i) + '.jpg')


def filter_data(images, labels, singleLabel):
    x = (labels == singleLabel)
    y = x.nonzero()
    nums = images[y]
    image = Image.new('L', (28, 28))
    image.putdata(list(nums[random.randint(0, nums.size()[0])].view(-1)))
    # image.show()
    image.save(os.path.join('./filter_data/' + str(singleLabel) + '.jpg'))
    return nums


def merge_images(images, operation, number):
    labels = read_idx('train-labels.idx1-ubyte')
    x = (labels == number)
    y = x.nonzero()
    if operation == "max":
        nums = images[y]
        max = torch.max(nums)
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[max.type(torch.int32)].view(-1)))
        # ximage.show()
        image.save(os.path.join('./max/' + str(number) + '.jpg'))
    elif operation == "median":
        nums = images[y]
        media = torch.median(nums)
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[media.type(torch.int32)].view(-1)))
        # image.show()
        image.save(os.path.join('./median/' + str(number) + '.jpg'))
    elif operation == "mean":
        nums = images[y]
        mean = torch.mean(nums.type(torch.float32))
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[mean.type(torch.int32)].view(-1)))
        # image.show()
        image.save(os.path.join('./mean/' + str(number) + '.jpg'))
    else:
        print("Error! Operation not found")


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 197)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(197, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


def training_nonlin(a,x,b,c):
    y=a*(math.e**(b*x)) + c
    return y


def loss_fn(y, oov):
    dif = ((oov-y)**2).mean()
    return dif


def dmodel_w(y, w, b):
    return y


def dmodel_b(y, w, b):
    return 1.0


def dloss_m(y, oov):
    dsq_diffs = 2 * (y-oov)
    return dsq_diffs


def training_LinearWithTorch(n, alpha, x, y):
    t_p =nn.Linear(1, 1)
    loss_fn=nn.MSELoss()
    print ("TENSOR",)
    optimizer = optim.Adam(t_p.parameters(), lr=0.5)
    for i in range(0, n):
        loss = loss_fn(list(t_p.parameters())[0], y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d, Loss %f' % (i, float(loss)))


def training_opt(n, model, images, labels, optimizer):
    labels = labels.long()
    loss_fn = nn.CrossEntropyLoss()
    images = images.float().view(-1, 784)
    for i in range(0, n):
        t_p = model(images)
        loss = loss_fn(t_p, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d, Loss %f' % (i, float(loss)))

    torch.save(model.state_dict(), './save/nn')


if __name__ == '__main__':
    images = read_idx('train-images.idx3-ubyte')
    labels = read_idx('train-labels.idx1-ubyte')
    model = Network()
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    #training_opt(100, model, images, labels, optimizer)

    imagesTest = read_idx('t10k-images.idx3-ubyte')
    labelsTest = read_idx('t10k-labels.idx1-ubyte')

    model.load_state_dict(torch.load('./save/nn'))
    model.eval()
    pred = model(imagesTest.float().view(-1, 784))
    print(pred.max(1)[0], pred.max(1)[1])
    cm = confusion_matrix(labelsTest, pred.max(1)[1])
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cr = classification_report(labelsTest, pred.max(1)[1], target_names=target_names)
    print(cr)

    one = model.hidden.weight
    image = Image.new('L', (28, 28))
    for i in range(0, one.size()[0]):
        image.putdata(list(one[i].view(-1)))  # el -1 convierte a una dimension
        #image.show()
        image.save(os.path.join('./images/'+str(i) + '.jpg'))