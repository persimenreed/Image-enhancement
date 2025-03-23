import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

"""
Pretrained model collected from https://github.com/yjn870/SRCNN-pytorch
"""

# Simple CNN architecture corresponding to the original paper. We only operate on one channel, as SRCNN uses luminance
# The pretrained model is a 9-5-5. Features are extracted from a 9x9 kernel, mapped with a 5x5 kernel and reconstructed with another 5x5 kernel
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


def SRCNN_upscale(img, model_path='srcnn_x2.pth', scale_factor=2):
    """
    This function performs the enhancing of the image. The LR image, weights and a scale factor is used as input.
    """
    model = SRCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Image is converted to YCbCr. This is because SRCNN works only on the luminance channel
    img = img.convert('YCbCr')
    y, cb, cr = img.split()
    # SRCNN does not upscale the images, it used Bicubic interpolation for this. SRCNN improved already scaled images
    img_y_upscaled = y.resize((y.width * scale_factor, y.height * scale_factor), Image.BICUBIC)
    # Converts the channel to correct model input
    input_y = ToTensor()(img_y_upscaled).unsqueeze(0)

    with torch.no_grad():
        output = model(input_y)
    output = output.clamp(0.0, 1.0)
    # Converts back to PIL
    out_y_image = ToPILImage()(output.squeeze(0))

    # Upscaled the two remaining channels using the same Bicubic interpolation
    cb_up = cb.resize(out_y_image.size, Image.BICUBIC)
    cr_up = cr.resize(out_y_image.size, Image.BICUBIC)
    # Converting back to RGB
    result_img = Image.merge('YCbCr', (out_y_image, cb_up, cr_up)).convert('RGB')
    return result_img
