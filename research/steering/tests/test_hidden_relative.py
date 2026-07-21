import torch

from research.steering.experiments.reflexion_T.experimental.hidden_relative import HiddenRelativeSteerer


class Block(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block()])


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()


def test_hidden_relative_is_generation_only_and_norm_scaled():
    model = Toy()
    with HiddenRelativeSteerer(
        model=model,
        layer=0,
        vector=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        natural_rho=0.5,
        multiplier=1.0,
        min_addition_norm=0.0,
        max_addition_norm=10.0,
    ):
        prefill = model.model.layers[0](torch.ones(1, 2, 4))
        decode = model.model.layers[0](torch.ones(1, 1, 4))
    assert torch.allclose(prefill, torch.ones_like(prefill))
    # ||h||=2, rho=.5, so a unit direction receives addition norm 1.
    assert torch.allclose(decode, torch.tensor([[[2.0, 1.0, 1.0, 1.0]]]))
