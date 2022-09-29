from turtle import forward
from unicodedata import name
from matplotlib.pyplot import axis, cla
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image

from utils.dataloader import yolo_dataset_collate, IanDataset, YoloDataset, MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class Scale_bias(nn.Module):
    def __init__(self, gamma_init=1,beta_init=0, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        # super(Scale_bias, self).__init__(**kwargs)

    '''def build(self, input_shape):
        # self.input_spec = [InputSpec(shape=input_shape)]
        # gamma = self.gamma_init * np.ones(1)
        # beta = self.beta_init * np.ones(1)
        gamma = self.gamma_init
        beta = self.beta_init
        # self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        # self.beta = K.variable(beta, name='{}_beta'.format(self.name))
        self.gamma = nn.parameter(gamma)
        self.beta = nn.parameter(beta)
        self.trainable_weights = [self.gamma,self.beta]
        super(Scale_bias, self).build(input_shape)'''

    def forward(self, x, mask=None):
        # output = K.l2_normalize(x, self.axis)
        # x_pre = K.dot(x, self.gamma)
        # output = K.bias_add(x_pre, self.beta)
        # x_pre = x*self.gamma
        # output = x_pre+self.beta
        x_pre = torch.mul(x,self.gamma)
        output = torch.add(x_pre,self.beta)

        return output

class illumination(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_images = Lambda(lambda x: F.interpolate(x, [56, 56], mode='bilinear'))
        self.div = Lambda(lambda x: x / 255)
        self.original1 = Lambda(lambda x: x[:, :, :, 0] + 103.939)
        self.original2 = Lambda(lambda x: x[:, :, :, 1] + 116.779)
        self.original3 = Lambda(lambda x: x[:, :, :, 2] + 123.68)
        self.expand_dims = Lambda(lambda x: torch.unsqueeze(x, -1))

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.MaxPooling = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.dense1 = nn.Linear(6272, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 2)
        self.dense4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

        self.half_add = Lambda(lambda x: 0.5 + x)
        self.sub = Lambda(lambda x: (x[0] - x[1])*0.5)

        self.scale_bias = Scale_bias()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

    def forward(self, input_tensor_rgb):
        # normalize the input
        '''img_input_rgb1 = self.original1(input_tensor_rgb)
        img_input_rgb1 = self.expand_dims(img_input_rgb1)
        img_input_rgb2 = self.original2(input_tensor_rgb)
        img_input_rgb2 = self.expand_dims(img_input_rgb2)88
        img_input_rgb3 = self.original3(input_tensor_rgb)
        img_input_rgb3 = self.expand_dims(img_input_rgb3)
        img_input_rgb_pre = torch.cat([img_input_rgb1, img_input_rgb2, img_input_rgb3], axis = -1)'''

        img_input_concat_resize = self.resize_images(input_tensor_rgb)
        '''img_input_concat_resize = self.div(img_input_concat_resize)'''

        # predict the w_n,w_d
        img_input_concat_stage1 = self.conv1(img_input_concat_resize)
        img_input_concat_stage1 = F.relu(self.bn1(img_input_concat_stage1))
        img_input_concat_stage1 = self.MaxPooling(img_input_concat_stage1)

        img_input_concat_stage2 = self.conv2(img_input_concat_stage1)
        img_input_concat_stage2 = F.relu(self.bn2(img_input_concat_stage2))
        img_input_concat_stage2 = self.MaxPooling(img_input_concat_stage2)
        img_input_concat_stage2 = img_input_concat_stage2.view(-1, self.num_flat_features(img_input_concat_stage2))

        img_input_concat_dense = F.relu(self.dense1(img_input_concat_stage2))
        img_input_concat_dense = self.dropout(img_input_concat_dense)
        img_input_concat_dense = F.relu(self.dense2(img_input_concat_dense))
        img_input_concat_dense = self.dropout(img_input_concat_dense)
        w_n = F.relu(self.dense3(img_input_concat_dense))[:, :1]
        w_d = F.relu(self.dense3(img_input_concat_dense))[:, 1:]
        illuminate_output = torch.cat([w_n, w_d], axis = 1)
        # illuminate_output = F.relu(self.dense3(img_input_concat_dense))

        w_n_weight = F.sigmoid(w_n)  # LWIR
        w_d_weight = F.sigmoid(w_d)  # RGB

        # predict the w_absolute(|w|)
        img_input_concat_stage22 = self.conv2(img_input_concat_stage1)
        img_input_concat_stage22 = F.relu(self.bn2(img_input_concat_stage22))
        img_input_concat_stage22 = self.MaxPooling(img_input_concat_stage22)
        img_input_concat_stage22 = img_input_concat_stage22.view(-1, self.num_flat_features(img_input_concat_stage22))

        img_input_concat_dense_alf = F.sigmoid(self.dense1(img_input_concat_stage22))
        img_input_concat_dense_alf = self.dropout(img_input_concat_dense_alf)
        img_input_concat_dense_alf = F.sigmoid(self.dense2(img_input_concat_dense_alf))
        img_input_concat_dense_alf = self.dropout(img_input_concat_dense_alf)
        w_absolute = F.sigmoid(self.dense4(img_input_concat_dense_alf))

        illuminate_aware_alf_value = self.scale_bias(w_absolute)

        # the final illumination weight
        w_n_illuminate = F.tanh(w_n)  # LWIR
        w_d_illuminate = F.tanh(w_d)  # RGB
        illuminate_rgb_positive = self.sub([w_d_illuminate,w_n_illuminate])
        # illuminate_rgb_positive = self.half_add(w_n_illuminate)
        illuminate_aware_alf_pre = torch.mul(illuminate_rgb_positive, illuminate_aware_alf_value)
        w_rgb = self.half_add(illuminate_aware_alf_pre)

        return illuminate_output, w_n_weight, w_d_weight, w_rgb

def Illumination_Gate(output_rgb,output_lwir,w_rgb):
    stage_rgb = torch.mul(output_rgb,w_rgb)
    stage_lwir = torch.mul(output_lwir,1-w_rgb)
    print(stage_rgb)
    print(stage_lwir)
    return stage_rgb,stage_lwir

#kaist:24 FLIR:27
def fusion(rgb,lwir,w_rgb):
    output_rgb = (torch.zeros([len(rgb[0]), 24, 13, 13], dtype=torch.float).cuda(), torch.zeros([len(rgb[0]), 24, 26, 26], dtype=torch.float).cuda(), torch.zeros([len(rgb[0]), 24, 52, 52], dtype=torch.float).cuda())
    output_lwir = (torch.zeros([len(lwir[0]), 24, 13, 13], dtype=torch.float).cuda(), torch.zeros([len(lwir[0]), 24, 26, 26], dtype=torch.float).cuda(), torch.zeros([len(lwir[0]), 24, 52, 52], dtype=torch.float).cuda())
    for i, x in enumerate(rgb):
        for j, y in enumerate(x):
            output_rgb[i][j] = torch.mul(y, w_rgb[j])

    for i, x in enumerate(lwir):
        for j, y in enumerate(x): #x[z] for z in range(x.size(0)) => torch 1.7 error solution
            output_lwir[i][j] = torch.mul(y, 1-w_rgb[j])

    output_0 = torch.add(output_rgb[0], output_lwir[0])
    output_1 = torch.add(output_rgb[1], output_lwir[1])
    output_2 = torch.add(output_rgb[2], output_lwir[2])

    return output_0, output_1, output_2

# if __name__ == "__main__":
    
#     visible_train_annotation_path   = 'visible_train.txt'
#     lwir_train_annotation_path      = 'lwir_train.txt'
#     ian_train_annotation_path = 't.txt'

#     with open(visible_train_annotation_path) as f:
#         visible_train_lines = f.readlines()
#     with open(lwir_train_annotation_path) as f:
#         lwir_train_lines = f.readlines()
#     with open(ian_train_annotation_path) as f:
#         ian_train_lines = f.readlines()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     IAN = illumination()
#     IAN = IAN.train().to(device)
#     ian_loss = torch.nn.CrossEntropyLoss().to(device)

#     ian_optimizer   = optim.Adam(IAN.parameters(), 0.001)


#     visible_train_dataset   = YoloDataset(visible_train_lines, [640, 512], 3, mosaic=False, train = False)
#     lwir_train_dataset      = YoloDataset(lwir_train_lines, [640, 512], 3, mosaic=False, train = False)
#     IAN_train_dataset       = IanDataset(ian_train_lines)
#     myDataset_train = MyDataset(dataset1 = visible_train_dataset, dataset2 = lwir_train_dataset, dataset3 = IAN_train_dataset)
#     gen                     = DataLoader(myDataset_train, shuffle = True, batch_size = 8, num_workers = 4, pin_memory=True,
#                                     drop_last=True, collate_fn = yolo_dataset_collate)

#     for iteration, (visible_batch, lwir_batch, ian_batch) in enumerate(gen):
#         train_loss_ian = 0

#         visible_images, targets = visible_batch[0], visible_batch[1]
#         ian_images, labels = ian_batch[0], ian_batch[1]
#         visible_images  = torch.from_numpy(visible_images).type(torch.FloatTensor).cuda()
#         ian_images  = torch.from_numpy(ian_images).type(torch.FloatTensor)
#         labels = torch.tensor([torch.from_numpy(ann).type(torch.FloatTensor) for ann in labels])

#         ian_optimizer.zero_grad()
#         ian = IAN.forward(ian_images.to(device))
#         loss_ian = ian_loss(ian[0], labels.to(device, dtype=torch.int64))
#         train_loss_ian = train_loss_ian + loss_ian
#         loss_ian.backward()
#         ian_optimizer.step()

#         print(ian[0])
#         print(ian[1])
#         print(ian[2])
#         print(ian[3])
    