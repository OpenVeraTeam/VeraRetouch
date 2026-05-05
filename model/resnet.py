import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
import torch.nn.functional as F

class Resnet18Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Resnet18Encoder, self).__init__()
        self.model0 = Sequential(
            # 0
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP Adaptive Average Pooling
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)
        
        self.proj = Linear(1024, latent_dim*3)


    def forward(self, x1, x2):
        # x = self.model0(x)
        x1 = self.model0(x1)
        x2 = self.model0(x2)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.R1(x1)
        f1 = x2
        x2 = self.model1(x2)
        x2 = x2 + f1
        x2 = self.R1(x2)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.R2(x1)
        f1_1 = x2
        x2 = self.model2(x2)
        x2 = x2 + f1_1
        x2 = self.R2(x2)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.R3(x1)
        f2_1 = x2
        f2_1 = self.en1(f2_1)
        x2 = self.model3(x2)
        x2 = x2 + f2_1
        x2 = self.R3(x2)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)
        f2_2 = x2
        x2 = self.model4(x2)
        x2 = x2 + f2_2
        x2 = self.R4(x2)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.R5(x1)
        f3_1 = x2
        f3_1 = self.en2(f3_1)
        x2 = self.model5(x2)
        x2 = x2 + f3_1
        x2 = self.R5(x2)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)
        f3_2 = x2
        x2 = self.model6(x2)
        x2 = x2 + f3_2
        x2 = self.R6(x2)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.R7(x1)
        f4_1 = x2
        f4_1 = self.en3(f4_1)
        x2 = self.model7(x2)
        x2 = x2 + f4_1
        x2 = self.R7(x2)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)
        f4_2 = x1
        x2 = self.model8(x2)
        x2 = x2 + f4_2
        x2 = self.R8(x2)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        x2 = self.aap(x2)
        x2 = self.flatten(x2)
        x2 = self.fc(x2)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat

class Resnet22Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Resnet22Encoder, self).__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        self.model9 = Sequential(
            # 5.1
            Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(1024),
            ReLU(),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(),
        )
        self.en4 = Sequential(
            Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(1024),
            ReLU(),
        )
        self.R9 = ReLU()

        self.model10 = Sequential(
            # 5.2
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(),
        )
        self.R10 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(1024, 1024)
        
        self.proj = Linear(2048, latent_dim*3)


    def forward(self, x1, x2):
        # x = self.model0(x)
        x1 = self.model0(x1)
        x2 = self.model0(x2)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.R1(x1)
        f1 = x2
        x2 = self.model1(x2)
        x2 = x2 + f1
        x2 = self.R1(x2)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.R2(x1)
        f1_1 = x2
        x2 = self.model2(x2)
        x2 = x2 + f1_1
        x2 = self.R2(x2)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.R3(x1)
        f2_1 = x2
        f2_1 = self.en1(f2_1)
        x2 = self.model3(x2)
        x2 = x2 + f2_1
        x2 = self.R3(x2)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)
        f2_2 = x2
        x2 = self.model4(x2)
        x2 = x2 + f2_2
        x2 = self.R4(x2)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.R5(x1)
        f3_1 = x2
        f3_1 = self.en2(f3_1)
        x2 = self.model5(x2)
        x2 = x2 + f3_1
        x2 = self.R5(x2)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)
        f3_2 = x2
        x2 = self.model6(x2)
        x2 = x2 + f3_2
        x2 = self.R6(x2)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.R7(x1)
        f4_1 = x2
        f4_1 = self.en3(f4_1)
        x2 = self.model7(x2)
        x2 = x2 + f4_1
        x2 = self.R7(x2)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)
        f4_2 = x1
        x2 = self.model8(x2)
        x2 = x2 + f4_2
        x2 = self.R8(x2)

        f5_1 = x1
        f5_1 = self.en4(f5_1)
        x1 = self.model9(x1)
        x1 = x1 + f5_1
        x1 = self.R9(x1)
        f5_1 = x2
        f5_1 = self.en4(f5_1)
        x2 = self.model9(x2)
        x2 = x2 + f5_1
        x2 = self.R9(x2)

        f5_2 = x1
        x1 = self.model10(x1)
        x1 = x1 + f5_2
        x1 = self.R10(x1)
        f5_2 = x1
        x2 = self.model10(x2)
        x2 = x2 + f5_2
        x2 = self.R10(x2)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        x2 = self.aap(x2)
        x2 = self.flatten(x2)
        x2 = self.fc(x2)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat

class Resnet18_InteractEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(Resnet18_InteractEncoder, self).__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )

        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)

        self.proj = Linear(1024, latent_dim*3)

    def forward(self, x1, x2):
        # x = self.model0(x)
        x1 = self.model0(x1)
        x2 = self.model0(x2)
        
        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_


        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.R1(x1)
        f1 = x2
        x2 = self.model1(x2)
        x2 = x2 + f1
        x2 = self.R1(x2)

        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.R2(x1)
        f1_1 = x2
        x2 = self.model2(x2)
        x2 = x2 + f1_1
        x2 = self.R2(x2)

        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.R3(x1)
        f2_1 = x2
        f2_1 = self.en1(f2_1)
        x2 = self.model3(x2)
        x2 = x2 + f2_1
        x2 = self.R3(x2)
        
        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)
        f2_2 = x2
        x2 = self.model4(x2)
        x2 = x2 + f2_2
        x2 = self.R4(x2)

        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.R5(x1)
        f3_1 = x2
        f3_1 = self.en2(f3_1)
        x2 = self.model5(x2)
        x2 = x2 + f3_1
        x2 = self.R5(x2)

        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)
        f3_2 = x2
        x2 = self.model6(x2)
        x2 = x2 + f3_2
        x2 = self.R6(x2)

        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.R7(x1)
        f4_1 = x2
        f4_1 = self.en3(f4_1)
        x2 = self.model7(x2)
        x2 = x2 + f4_1
        x2 = self.R7(x2)

        x2_ = x2
        x2 = x2 - x1
        x1 = x1 - x2_

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)
        f4_2 = x1
        x2 = self.model8(x2)
        x2 = x2 + f4_2
        x2 = self.R8(x2)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        x2 = self.aap(x2)
        x2 = self.flatten(x2)
        x2 = self.fc(x2)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat


class CrossDifferenceModule(nn.Module):
    """Pixel-level difference extraction module"""
    def __init__(self, channels):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
    
    def forward(self, x1, x2):
        # Calculate bidirectional differences
        diff1 = torch.abs(x1 - x2)
        diff2 = torch.abs(x2 - x1)
        
        diff = torch.cat([diff1, diff2], dim=1)
        attn = self.diff_conv(diff)
        
        x1_out = x1 * attn + x1
        x2_out = x2 * (1 - attn) + x2
        return x1_out, x2_out

class Resnet18_CrossInteractEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Keep original ResNet structure unchanged...
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )

        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)

        # Add difference extraction modules
        self.diff_module1 = CrossDifferenceModule(64)   # after model1
        self.diff_module2 = CrossDifferenceModule(128)  # after model3
        self.diff_module3 = CrossDifferenceModule(256)  # after model5
        self.diff_module4 = CrossDifferenceModule(512)  # after model7
        
        # Enhance final feature fusion
        self.proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim*3)
        )
    
    def forward(self, x1, x2):
        # Initial feature extraction
        x1 = self.model0(x1)
        x2 = self.model0(x2)
        
        # === First layer block ===
        # Branch 1 processing
        f1_x1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1_x1
        x1 = self.R1(x1)
        
        # Branch 2 processing
        f1_x2 = x2
        x2 = self.model1(x2)
        x2 = x2 + f1_x2
        x2 = self.R1(x2)
        
        # Pixel-level difference extraction 1
        x1, x2 = self.diff_module1(x1, x2)
        
        # === Second layer block ===
        # Branch 1 processing
        f2_x1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f2_x1
        x1 = self.R2(x1)
        
        # Branch 2 processing
        f2_x2 = x2
        x2 = self.model2(x2)
        x2 = x2 + f2_x2
        x2 = self.R2(x2)
        
        # === Third layer block (with downsampling) ===
        # Branch 1 processing
        f3_x1 = self.en1(x1)  # Downsampling shortcut
        x1 = self.model3(x1)
        x1 = x1 + f3_x1
        x1 = self.R3(x1)
        
        # Branch 2 processing
        f3_x2 = self.en1(x2)  # Downsampling shortcut
        x2 = self.model3(x2)
        x2 = x2 + f3_x2
        x2 = self.R3(x2)
        
        # Pixel-level difference extraction 2
        x1, x2 = self.diff_module2(x1, x2)
        
        # === Fourth layer block ===
        # Branch 1 processing
        f4_x1 = x1
        x1 = self.model4(x1)
        x1 = x1 + f4_x1
        x1 = self.R4(x1)
        
        # Branch 2 processing
        f4_x2 = x2
        x2 = self.model4(x2)
        x2 = x2 + f4_x2
        x2 = self.R4(x2)
        
        # === Fifth layer block (with downsampling) ===
        # Branch 1 processing
        f5_x1 = self.en2(x1)  # Downsampling shortcut
        x1 = self.model5(x1)
        x1 = x1 + f5_x1
        x1 = self.R5(x1)
        
        # Branch 2 processing
        f5_x2 = self.en2(x2)  # Downsampling shortcut
        x2 = self.model5(x2)
        x2 = x2 + f5_x2
        x2 = self.R5(x2)
        
        # Pixel-level difference extraction 3
        x1, x2 = self.diff_module3(x1, x2)
        
        # === Sixth layer block ===
        # Branch 1 processing
        f6_x1 = x1
        x1 = self.model6(x1)
        x1 = x1 + f6_x1
        x1 = self.R6(x1)
        
        # Branch 2 processing
        f6_x2 = x2
        x2 = self.model6(x2)
        x2 = x2 + f6_x2
        x2 = self.R6(x2)
        
        # === Seventh layer block (with downsampling) ===
        # Branch 1 processing
        f7_x1 = self.en3(x1)  # Downsampling shortcut
        x1 = self.model7(x1)
        x1 = x1 + f7_x1
        x1 = self.R7(x1)
        
        # Branch 2 processing
        f7_x2 = self.en3(x2)  # Downsampling shortcut
        x2 = self.model7(x2)
        x2 = x2 + f7_x2
        x2 = self.R7(x2)
        
        # Pixel-level difference extraction 4
        x1, x2 = self.diff_module4(x1, x2)
        
        # === Eighth layer block ===
        # Branch 1 processing
        f8_x1 = x1
        x1 = self.model8(x1)
        x1 = x1 + f8_x1
        x1 = self.R8(x1)
        
        # Branch 2 processing
        f8_x2 = x2
        x2 = self.model8(x2)
        x2 = x2 + f8_x2
        x2 = self.R8(x2)
        
        # === Feature extraction and fusion ===
        # Adaptive average pooling
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        
        x2 = self.aap(x2)
        x2 = self.flatten(x2)
        x2 = self.fc(x2)
        
        # Concatenate features and project
        feat = torch.cat([x1, x2], dim=-1)
        return self.proj(feat)

# Spatial Attention Module (Lightweight)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        att_map = self.sigmoid(self.conv(concat))
        return x * att_map

# Channel Attention Module (SE Block)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        channel_att = self.channel_att(x)
        x_channel = x * channel_att
        
        # Spatial attention
        max_pool = torch.max(x_channel, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_att = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_att(spatial_att)
        x_spatial = x_channel * spatial_att
        
        return x_spatial


class Resnet18Encoder_CBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)
        
        self.proj = Linear(1024, latent_dim*3)
        
        # Add CBAM modules
        self.cbam1 = CBAM(64)    # for model1/model2 output
        self.cbam2 = CBAM(128)   # for model3/model4 output
        self.cbam3 = CBAM(256)   # for model5/model6 output
        self.cbam4 = CBAM(512)   # for model7/model8 output

        # ... [remaining layers unchanged] ...

    def forward_x(self, x1):
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.cbam1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.cbam2(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.cbam3(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.cbam4(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1
    
    def forward(self, x1, x2):
        x1 = self.forward_x(x1)
        x2 = self.forward_x(x2)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat

class Resnet18Encoder_MixedCBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)
        
        self.proj = Linear(1024, latent_dim*3)
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels

        self.attn_deep = ChannelAttention(512)   # 512channels


    def forward_x(self, x1):
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_deep(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1
    
    def forward(self, x1, x2):
        x1 = self.forward_x(x1)
        x2 = self.forward_x(x2)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat

class Resnet18Encoder_MinusMixedCBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 1024)
        
        self.proj = Linear(1024, latent_dim*3)
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels

        self.attn_deep = ChannelAttention(512)   # 512channels


    def forward_x(self, x1):
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_deep(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1
    
    def forward(self, x1, x2):
        x1 = self.forward_x(x1)
        x2 = self.forward_x(x2)
        feat = x2 - x1
        feat = self.proj(feat)
        return feat


class Resnet18Encoder_MinusMixedCBAMBaseBlock(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 1024)
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels

        self.attn_deep = ChannelAttention(512)   # 512channels


    def forward(self, x1):
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_deep(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1

class Resnet18Encoder_2NetMinusMixedCBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net_1 = Resnet18Encoder_MinusMixedCBAMBaseBlock(latent_dim)
        self.net_2 = Resnet18Encoder_MinusMixedCBAMBaseBlock(latent_dim)
        self.proj = Linear(1024, latent_dim*3)
        
    def forward(self, x1, x2):
        x1 = self.net_1(x1)
        x2 = self.net_2(x2)
        feat = x2 - x1
        feat = self.proj(feat)
        return feat

class Resnet18Encoder_SpecialAtten(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)
        
        self.proj = Linear(1024, latent_dim*3)
        
        # add CBAM module
        self.attn_spatial0 = SpatialAttention()  # 64 channels
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels


    def forward_x(self, x1):
        x1 = self.model0(x1)
        x1 = self.attn_spatial0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1
    
    def forward(self, x1, x2):
        x1 = self.forward_x(x1)
        x2 = self.forward_x(x2)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat


class Resnet18Encoder_InputRes(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=6, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, 512)
        
        self.proj = Linear(1024, latent_dim*3)
    

    def forward_x(self, x1):
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1
    
    def forward(self, x1, x2):
        x1_input = torch.cat([x1, x2 - x1], dim=1)
        x2_input = torch.cat([x2, x2 - x1], dim=1)
        x1 = self.forward_x(x1_input)
        x2 = self.forward_x(x2_input)
        feat = torch.cat([x1, x2], dim=-1)
        feat = self.proj(feat)
        return feat

class Resnet18Encoder_InputCatMixedCBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=6, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, latent_dim*3)
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels

        self.attn_deep = ChannelAttention(512)   # 512channels


    def forward(self, x1, x2):
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_deep(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1

class Resnet18VQEncoder_InputCatMixedCBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=6, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        self.model9 = Sequential(
            # 4.1
            Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(1024),
            ReLU(),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(),
        )
        self.en4 = Sequential(
            Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(1024),
            ReLU(),
        )
        self.R9 = ReLU()

        self.model10 = Sequential(
            # 4.2
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(),
        )
        self.R10 = ReLU()

        self.fc = Conv2d(in_channels=1024, out_channels=latent_dim * 3, kernel_size=(1, 1), stride=1, padding=0)
        
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels
        self.attn_mid3 = CBAM(512)   # 256channels

        self.attn_deep = ChannelAttention(1024)   # 512channels


    def forward(self, x1, x2):
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_mid3(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        f5_1 = x1
        f5_1 = self.en4(f5_1)
        x1 = self.model9(x1)
        x1 = x1 + f5_1
        x1 = self.attn_deep(x1)
        x1 = self.R9(x1)

        f5_2 = x1
        x1 = self.model10(x1)
        x1 = x1 + f5_2
        x1 = self.R10(x1)

        x1 = self.fc(x1)
        return x1
    

class Resnet18Encoder_3InputCatMixedCBAM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=9, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 
        self.flatten = Flatten(start_dim=1)
        # Fully connected
        self.fc = Linear(512, latent_dim*3)
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels

        self.attn_deep = ChannelAttention(512)   # 512channels


    def forward(self, x1, x2, x0):
        x1 = torch.cat([x0, x1, x2], dim=1)
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_deep(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)

        # last 3
        x1 = self.aap(x1)
        x1 = self.flatten(x1)
        x1 = self.fc(x1)
        return x1


class Resnet18Encoder_FPN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # Fully connected
        self.fc = Linear(512, 512)

        self.lateral_convs = nn.ModuleDict({
            'C2': Conv2d(64, 128, 1),
            'C3': Conv2d(128, 128, 1),
            'C4': Conv2d(256, 128, 1),
            'C5': Conv2d(512, 128, 1),
        })

        # 自顶向下融合后的卷积层（消除上采样的混叠效应）
        self.fpn_convs = nn.ModuleDict({
            'P2': Conv2d(128, 128, 3, padding=1),
            'P3': Conv2d(128, 128, 3, padding=1),
            'P4': Conv2d(128, 128, 3, padding=1),
            'P5': Conv2d(128, 128, 3, padding=1),
        })

        self.downsample = nn.Sequential(
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
            nn.AvgPool2d(4, stride=4),
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
            nn.AvgPool2d(4, stride=4),
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
            nn.AvgPool2d(4, stride=4),
            Conv2d(1024, 1024, 3, padding=1),
        )
        self.flatten = Flatten(start_dim=1)
        self.proj = Linear(1024, latent_dim*3)

    def forward_fpn(self, x):
        # 提取中间特征
        c_features = self.forward_x(x)
        
        # 横向连接
        p5 = self.lateral_convs['C5'](c_features['C5'])
        p4 = F.interpolate(p5, size=c_features['C4'].shape[-2:], mode='nearest') + self.lateral_convs['C4'](c_features['C4'])
        p3 = F.interpolate(p4, size=c_features['C3'].shape[-2:], mode='nearest') + self.lateral_convs['C3'](c_features['C3'])
        p2 = p3 + self.lateral_convs['C2'](F.interpolate(c_features['C2'], size=c_features['C3'].shape[-2:], mode='nearest'))
        
        # 最终特征图
        p_features = {
            'P2': self.fpn_convs['P2'](p2), # 64*64
            'P3': self.fpn_convs['P3'](p3), # 64*64
            'P4': self.fpn_convs['P4'](p4), # 32*32
            'P5': self.fpn_convs['P5'](p5), # 16*16
        }
        return p_features


    def feat_resize(self, p_features):
        p2 = p_features["P2"]
        p3 = p_features["P3"]
        p4 = F.interpolate(p_features["P4"], size=(64, 64), mode='nearest')
        p5 = F.interpolate(p_features["P5"], size=(64, 64), mode='nearest')
        p_features = {
            'P2': p2, 
            'P3': p3,
            'P4': p4,
            'P5': p5,
        }
        return p_features
    
    def forward_x(self, x1):
        outputs = {}
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.R2(x1)
        outputs['C2'] = x1

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)
        outputs['C3'] = x1

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)
        outputs['C4'] = x1

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)
        outputs['C5'] = x1

        return outputs
    
    def forward(self, x1, x2):
        fpn1 = self.forward_fpn(x1)
        fpn2 = self.forward_fpn(x2)
        fpn1 = self.feat_resize(fpn1)
        fpn2 = self.feat_resize(fpn2)
        feats = []
        for lvl in ['P2','P3','P4','P5']:
            f1 = fpn1[lvl]
            f2 = fpn2[lvl]
            feats.append(torch.cat([f1, f2], dim=1))
        fuse = torch.cat(feats, dim=1)
        fuse_feat = self.downsample(fuse)
        final_feat = self.flatten(fuse_feat)
        final_feat = torch.relu(final_feat)
        final_feat = self.proj(final_feat)
        return final_feat


class Resnet18Encoder_InputCatMixedCBAMFPN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model0 = Sequential(
            # 0
            
            Conv2d(in_channels=6, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()
        
        # add CBAM module
        self.attn_spatial1 = SpatialAttention()  # 64 channels
        self.attn_spatial2 = SpatialAttention()  # 64 channels

        self.attn_mid1 = CBAM(128)   # 128channels
        self.attn_mid2 = CBAM(256)   # 256channels

        self.attn_deep = ChannelAttention(512)   # 512channels

        self.lateral_convs = nn.ModuleDict({
            'C2': Conv2d(64, 256, 1),
            'C3': Conv2d(128, 256, 1),
            'C4': Conv2d(256, 256, 1),
            'C5': Conv2d(512, 256, 1),
        })

        # 自顶向下融合后的卷积层（消除上采样的混叠效应）
        self.fpn_convs = nn.ModuleDict({
            'P2': Conv2d(256, 256, 3, padding=1),
            'P3': Conv2d(256, 256, 3, padding=1),
            'P4': Conv2d(256, 256, 3, padding=1),
            'P5': Conv2d(256, 256, 3, padding=1),
        })

        self.downsample = nn.Sequential(
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
            nn.AvgPool2d(4, stride=4),
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
            nn.AvgPool2d(4, stride=4),
            Conv2d(1024, 1024, 3, padding=1),
            ReLU(),
            nn.AvgPool2d(4, stride=4),
            Conv2d(1024, 1024, 3, padding=1),
        )
        self.flatten = Flatten(start_dim=1)
        self.proj = Linear(1024, latent_dim*3)

    def forward_fpn(self, x1, x2):
        # 提取中间特征
        c_features = self.forward_x1_x2(x1, x2)
        
        # 横向连接
        p5 = self.lateral_convs['C5'](c_features['C5'])
        p4 = F.interpolate(p5, size=c_features['C4'].shape[-2:], mode='nearest') + self.lateral_convs['C4'](c_features['C4'])
        p3 = F.interpolate(p4, size=c_features['C3'].shape[-2:], mode='nearest') + self.lateral_convs['C3'](c_features['C3'])
        p2 = p3 + self.lateral_convs['C2'](F.interpolate(c_features['C2'], size=c_features['C3'].shape[-2:], mode='nearest'))
        
        # 最终特征图
        p_features = {
            'P2': self.fpn_convs['P2'](p2), # 64*64
            'P3': self.fpn_convs['P3'](p3), # 64*64
            'P4': self.fpn_convs['P4'](p4), # 32*32
            'P5': self.fpn_convs['P5'](p5), # 16*16
        }
        return p_features


    def feat_resize(self, p_features):
        p2 = p_features["P2"]
        p3 = p_features["P3"]
        p4 = F.interpolate(p_features["P4"], size=(64, 64), mode='nearest')
        p5 = F.interpolate(p_features["P5"], size=(64, 64), mode='nearest')
        p_features = {
            'P2': p2, 
            'P3': p3,
            'P4': p4,
            'P5': p5,
        }
        return p_features

    def forward_x1_x2(self, x1, x2):
        outputs = {}
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.model0(x1)

        f1 = x1
        x1 = self.model1(x1)
        x1 = x1 + f1
        x1 = self.attn_spatial1(x1)
        x1 = self.R1(x1)
        
        f1_1 = x1
        x1 = self.model2(x1)
        x1 = x1 + f1_1
        x1 = self.attn_spatial2(x1)
        x1 = self.R2(x1)
        outputs['C2'] = x1

        f2_1 = x1
        f2_1 = self.en1(f2_1)
        x1 = self.model3(x1)
        x1 = x1 + f2_1
        x1 = self.attn_mid1(x1)
        x1 = self.R3(x1)

        f2_2 = x1
        x1 = self.model4(x1)
        x1 = x1 + f2_2
        x1 = self.R4(x1)
        outputs['C3'] = x1

        f3_1 = x1
        f3_1 = self.en2(f3_1)
        x1 = self.model5(x1)
        x1 = x1 + f3_1
        x1 = self.attn_mid2(x1)
        x1 = self.R5(x1)

        f3_2 = x1
        x1 = self.model6(x1)
        x1 = x1 + f3_2
        x1 = self.R6(x1)
        outputs['C4'] = x1

        f4_1 = x1
        f4_1 = self.en3(f4_1)
        x1 = self.model7(x1)
        x1 = x1 + f4_1
        x1 = self.attn_deep(x1)
        x1 = self.R7(x1)

        f4_2 = x1
        x1 = self.model8(x1)
        x1 = x1 + f4_2
        x1 = self.R8(x1)
        outputs['C5'] = x1

        return outputs

    def forward(self, x1, x2):
        fpn = self.forward_fpn(x1, x2)
        fpn = self.feat_resize(fpn)
        feats = []
        for lvl in ['P2','P3','P4','P5']:
            f = fpn[lvl]
            feats.append(f)
        fuse = torch.cat(feats, dim=1)
        fuse_feat = self.downsample(fuse)
        final_feat = self.flatten(fuse_feat)
        final_feat = torch.relu(final_feat)
        final_feat = self.proj(final_feat)
        return final_feat

if __name__ == "__main__":
    model = Resnet18VQEncoder_InputCatMixedCBAM(latent_dim=896)
    x1 = torch.rand(4,3,512,512)
    x2 = torch.rand(4,3,512,512)
    feat = model(x1, x2)
    print(feat.shape)