import torch.nn as nn

from Model.utils import MLP, ResidualBlock


class MutualNet(nn.Module):
    def __init__(self, opt):
        super(MutualNet, self).__init__()
        self.planes = opt.num_filters * 4
        self.MLP = MLP(8, self.planes, 4)
        self.model_start = nn.Sequential(
            nn.Conv2d(1, self.planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=self.planes, num_groups=1, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer0 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer1 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer2 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer3 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.model_end = nn.Sequential(
            nn.Conv2d(self.planes, opt.num_filters, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(opt.num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def _split_film(self, feature_vec, batch):
        if feature_vec.dim() == 1:
            feature_vec = feature_vec.unsqueeze(0)
        gamma, beta = self.MLP(feature_vec)
        params = []
        for idx in range(4):
            start = idx * self.planes
            end = (idx + 1) * self.planes
            params.append(
                (
                    gamma[:, start:end].view(batch, self.planes, 1, 1),
                    beta[:, start:end].view(batch, self.planes, 1, 1),
                )
            )
        return params

    def forward(self, input, feature_vec):
        batch = input.size(0)
        film = self._split_film(feature_vec, batch)
        output = self.model_start(input)
        output = film[0][0] * self.layer0(output) + film[0][1]
        output = film[1][0] * self.layer1(output) + film[1][1]
        output = film[2][0] * self.layer2(output) + film[2][1]
        output = film[3][0] * self.layer3(output) + film[3][1]
        return self.model_end(output)


MutualNet_f = MutualNet
