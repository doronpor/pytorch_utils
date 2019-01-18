# pytorch_utils
This repository contains pytorch utilities.

Requirements: Pytorch 1.0, torchvision


## Exmaples

profiler:

profile the [resnet18](https://arxiv.org/pdf/1512.03385.pdf) network
```python
from pytorch_utils.profiler import profile_modules
from torchvision.models.resnet import ResNet, BasicBlock

# wrap resnet18 with profiler
ResNet_profiling = profile_modules(enable=True, skip_first=True)(ResNet)

# init resnet18
network = ResNet_profiling(BasicBlock, [2, 2, 2, 2])

# move network to gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network.to(device)

input = torch.ones(1, 3, 224, 224, dtype=torch.float, device=device)

for i in range(10):
    network(input)

print(network)
```
output:
```
  Avg CPU Time  Avg GPU Time      hits       Total Time    Parameters      Input         FLOPS      Architecture 
=================================================================================================================
    4.79us        41.90us          9         420.23us       46.76MB       28.20MB      1.81Gmac    ResNet(
   145.11ns      721.07ns          9          7.80us        37.63KB      602.11KB     118.01Mmac     (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    66.68ns      356.39ns          9          3.81us        512.00        3.21MB           -         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    57.55ns       1.27us           9          11.97us          -          3.21MB           -         (relu): ReLU(inplace)
    88.08ns       1.51us           9          14.42us          -          3.21MB           -         (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
   897.35ns       6.28us           9          64.56us      591.87KB       8.03MB      462.42Mmac     (layer1): Sequential(
   442.82ns       3.12us           9          32.09us      295.94KB       4.01MB      231.21Mmac       (0): BasicBlock(
   142.00ns      648.11ns          9          7.11us       147.46KB      802.82KB     115.61Mmac         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    65.10ns      302.61ns          9          3.31us        512.00       802.82KB          -             (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    54.23ns       1.26us           19         24.95us          -         802.82KB          -             (relu): ReLU(inplace)
   125.45ns      622.24ns          9          6.73us       147.46KB      802.82KB     115.61Mmac         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    56.05ns      290.33ns          9          3.12us        512.00       802.82KB          -             (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
   454.53ns       3.15us           9          32.47us      295.94KB       4.01MB      231.21Mmac       (1): BasicBlock(
   136.16ns      626.88ns          9          6.87us       147.46KB      802.82KB     115.61Mmac         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    61.11ns      290.16ns          9          3.16us        512.00       802.82KB          -             (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    51.17ns       1.30us           19         25.63us          -         802.82KB          -             (relu): ReLU(inplace)
   143.04ns      630.84ns          9          6.96us       147.46KB      802.82KB     115.61Mmac         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    63.06ns      308.04ns          9          3.34us        512.00       802.82KB          -             (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
                                                                                                     )
    1.09us        7.60us           9          78.15us       2.10MB        5.62MB      411.04Mmac     (layer2): Sequential(
   631.32ns       4.01us           9          41.81us      920.58KB       3.61MB      179.83Mmac       (0): BasicBlock(
   133.55ns      560.73ns          9          6.25us       294.91KB      802.82KB      57.80Mmac         (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    61.42ns      305.32ns          9          3.30us        1.02KB       401.41KB          -             (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    55.60ns       1.29us           19         25.63us          -         401.41KB          -             (relu): ReLU(inplace)
   130.55ns      810.10ns          9          8.47us       589.82KB      401.41KB     115.61Mmac         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    58.41ns      293.44ns          9          3.17us        1.02KB       401.41KB          -             (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   191.78ns      751.68ns          9          8.49us        33.79KB       1.20MB       6.42Mmac          (downsample): Sequential(
   132.06ns      460.35ns          9          5.33us        32.77KB      802.82KB      6.42Mmac            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    59.72ns      291.33ns          9          3.16us        1.02KB       401.41KB          -               (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                         )
                                                                                                       )
   455.25ns       3.58us           9          36.34us       1.18MB        2.01MB      231.21Mmac       (1): BasicBlock(
   127.01ns      826.52ns          9          8.58us       589.82KB      401.41KB     115.61Mmac         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    54.09ns      282.41ns          9          3.03us        1.02KB       401.41KB          -             (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    54.27ns       1.32us           19         26.20us          -         401.41KB          -             (relu): ReLU(inplace)
   147.84ns      820.03ns          9          8.71us       589.82KB      401.41KB     115.61Mmac         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    72.05ns      329.29ns          9          3.61us        1.02KB       401.41KB          -             (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
                                                                                                     )
    1.07us        8.46us           9          85.77us       8.40MB        2.81MB      411.04Mmac     (layer3): Sequential(
   632.85ns       4.41us           9          45.36us       3.68MB        1.81MB      179.83Mmac       (0): BasicBlock(
   151.69ns      737.26ns          9          8.00us        1.18MB       401.41KB      57.80Mmac         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    61.93ns      296.16ns          9          3.22us        2.05KB       200.70KB          -             (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    49.66ns       1.28us           19         25.32us          -         200.70KB          -             (relu): ReLU(inplace)
   125.92ns       1.05us           9          10.63us       2.36MB       200.70KB     115.61Mmac         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    59.91ns      295.65ns          9          3.20us        2.05KB       200.70KB          -             (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   183.74ns      740.19ns          9          8.32us       133.12KB      602.11KB      6.42Mmac          (downsample): Sequential(
   125.79ns      438.47ns          9          5.08us       131.07KB      401.41KB      6.42Mmac            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    57.95ns      301.72ns          9          3.24us        2.05KB       200.70KB          -               (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                         )
                                                                                                       )
   440.82ns       4.05us           9          40.40us       4.72MB        1.00MB      231.21Mmac       (1): BasicBlock(
   125.60ns       1.06us           9          10.71us       2.36MB       200.70KB     115.61Mmac         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    59.44ns      290.86ns          9          3.15us        2.05KB       200.70KB          -             (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    58.00ns       1.33us           19         26.33us          -         200.70KB          -             (relu): ReLU(inplace)
   134.83ns       1.06us           9          10.71us       2.36MB       200.70KB     115.61Mmac         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    62.96ns      310.58ns          9          3.36us        2.05KB       200.70KB          -             (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
                                                                                                     )
    1.24us        13.60us          9         133.52us       33.57MB       1.40MB      411.04Mmac     (layer4): Sequential(
   744.46ns       6.41us           9          64.41us       14.69MB      903.17KB     179.83Mmac       (0): BasicBlock(
   181.26ns      900.87ns          9          9.74us        4.72MB       200.70KB      57.80Mmac         (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    61.35ns      323.84ns          9          3.47us        4.10KB       100.35KB          -             (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    60.17ns       1.40us           19         27.82us          -         100.35KB          -             (relu): ReLU(inplace)
   148.29ns       2.59us           9          24.67us       9.44MB       100.35KB     115.61Mmac         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    67.92ns      334.68ns          9          3.62us        4.10KB       100.35KB          -             (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   225.46ns      856.32ns          9          9.74us       528.38KB      301.06KB      6.42Mmac          (downsample): Sequential(
   150.58ns      527.26ns          9          6.10us       524.29KB      200.70KB      6.42Mmac            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    74.88ns      329.07ns          9          3.64us        4.10KB       100.35KB          -               (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                         )
                                                                                                       )
   493.10ns       7.19us           9          69.11us       18.88MB      501.76KB     231.21Mmac       (1): BasicBlock(
   147.71ns       2.61us           9          24.81us       9.44MB       100.35KB     115.61Mmac         (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    78.03ns      329.12ns          9          3.66us        4.10KB       100.35KB          -             (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    56.03ns       1.33us           19         26.32us          -         100.35KB          -             (relu): ReLU(inplace)
   146.73ns       2.58us           9          24.58us       9.44MB       100.35KB     115.61Mmac         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    64.60ns      335.03ns          9          3.60us        4.10KB       100.35KB          -             (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
                                                                                                     )
    76.48ns       1.37us           9          13.03us          -         100.35KB          -         (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
    62.37ns      738.70ns          9          7.21us        2.05MB        2.05KB      513.00Kmac     (fc): Linear(in_features=512, out_features=1000, bias=True)
                                                                                                   )
```
