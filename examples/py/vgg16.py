import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

import mimir_frontend.backend


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Test code
batch_size = 10
num_classes = 1000


def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]


class FunctionalWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, param_names):
        super().__init__()
        # Keep a reference to the original module for calling its forward,
        # but do NOT register it as a submodule (so its real parameters
        # are not picked up / traced as constants). We store it in a plain
        # list to hide it from nn.Module's parameter/module registration.
        object.__setattr__(self, "_base_model", base_model)
        self.param_names = list(param_names)

    @torch.compile(
        backend="mimir",
        options={
            "debug_dir": f"{os.path.dirname(os.path.realpath(__file__))}/../mim_debug",
            "emit_mlir": True,
        },
    )
    def forward(self, x, *params):
        params_dict = dict(zip(self.param_names, params))
        return functional_call(self._base_model, params_dict, (x,))


if __name__ == "__main__":
    model = Model(num_classes=num_classes)
    model.eval()

    # Collect parameters in a fixed order (this order = the order of the
    # extra arguments the compiled MLIR function will expect).
    param_names = [name for name, _ in model.named_parameters()]
    param_values = tuple(p.detach() for _, p in model.named_parameters())

    x = get_inputs()[0]

    with torch.no_grad():
        want = model(x)
    print("PyTorch output:")
    print(want)

    wrapper = FunctionalWrapper(model, param_names)
    wrapper.eval()

    with torch.no_grad():
        got = wrapper(x, *param_values)
    assert torch.allclose(want, got), (
        "functional wrapper does not match original model!"
    )
    print("Functional wrapper matches original model output. Good.")
