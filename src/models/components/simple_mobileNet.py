import torch
from torch import nn
from torchvision.models import get_model, list_models, get_model_weights, get_weight

output_shape = [68, 2]
class simple_mobileNet(nn.Module):
    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        weights: str = "DEFAULT",
        output_shape: list = [68, 2]
    ):
        super().__init__()
        model = get_model(model_name, weights = weights)

        model.classifier[1] = torch.nn.Linear(1280, 68 * 2)

        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=output_shape[0]*output_shape[1])

        self.model = model
        self.output_shape = output_shape
    
    def forward(self, x):
        return self.model(x).reshape(x.size(0), self.output_shape[0], self.output_shape[1])

if __name__ == "__main__":
    # print(available_models)
    m = simple_mobileNet()
    output = m(torch.randn(1, 3, 224, 224))
    print(output.shape)
