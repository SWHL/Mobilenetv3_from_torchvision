#### Mobilenetv3 from torchvision
- 有时并不能采用最新的torchvision（0.12），但是想使用0.11才有的mv3模型，于是就简单整理了一下。

#### 环境要求
- PyTorch: >= 1.6

#### 预训练模型手动下载
- mobilenet_v3_large: https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
- mobilenet_v3_small: https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

#### 使用方法
```python
from mobilenetv3 import mobilenet_v3_large

model_path = 'pretrained_models/mobilenet_v3_large-8738ca79.pth'
large = mobilenet_v3_large(pretrained=model_path, width_mult=1.0,
                           reduced_tail=False, dilated=False)
print(large)
```
