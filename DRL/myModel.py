import gym
import torch as th
import torch.nn as nn
import timm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim, model_name):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        print('building ', model_name)
        self.cnn = timm.create_model(model_name, pretrained=False, in_chans=n_input_channels, num_classes=0)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class CustomCNN_mask_feat(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim, model_name):
        super(CustomCNN_mask_feat, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        print('building ', model_name)
        self.model = timm.create_model(model_name, pretrained=False, in_chans=n_input_channels, num_classes=0)
        self.feature_extractor = th.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        feat_map = self.feature_extractor(observations)
        big_feat_map = nn.functional.interpolate(feat_map, (64, 64), mode='nearest')
        # big_feat_map = feat_map
        idxmask = observations[:, -1]
        gfeat_list = []

        for i in range(idxmask.shape[0]):
            idximg = idxmask[i]
            idximg_r = nn.functional.interpolate(idximg[None, None], feat_map.shape[2], mode='nearest')[0, 0]
            tem = th.where(idximg_r != 0)
            lfeat = feat_map[i:i + 1, :, tem[0].min():tem[0].max(), tem[1].min():tem[1].max()]
            # import pdb; pdb.set_trace()
            # print(tem[0].min(),tem[0].max(), tem[1].min(),tem[1].max())
            out = lfeat.mean(dim=[2, 3])
            gfeat_list.append(out)
        gfeat = th.cat(gfeat_list, dim=0)
        # import pdb; pdb.set_trace()
        return gfeat

