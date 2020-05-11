import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from model_builder import ModelBuilder


def test_resnet():
    model_builder = ModelBuilder(10, pretrained=False)
    print(model_builder.available_models)
    model = model_builder['resnet50']
    print(model)


if __name__ == "__main__":
    test_resnet()