# Scale-aware Deep Reinforcement Learning for High Resolution Remote Sensing Imagery Classification

<span style="font-size: 18px;">
Tired of being restricted by limited network input sizes (e.g., 512x512) in high-resolution image segmentation? Let the network choose its own scale!
</span>

![fig](figs/Cover_v1.png)

## Getting Started
This repository contains the implementation of our project using `stable-baselines3==1.6` and `gym==0.21`. To ensure compatibility and optimal performance, please use these specific versions.

For users employing `stable-baselines3>=1.7`, adjustments to the Environment may be required for compatibility with `gymnasium`. Refer to the documentation of [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) and [gymnasium](https://gymnasium.farama.org/api/env/) for guidance.

## Model Download

Structure of `saved_model.zip`:

```
saved_model.zip/
├── data.json          - JSON file containing class parameters (dictionary format)
├── *.optimizer.pth    - Serialized PyTorch optimizers
├── policy.pth         - PyTorch state dictionary for the saved policy
├── pytorch_variables.pth - Additional PyTorch variables
├── version.txt        - Stable Baselines3 version used for model saving
├── system_info.txt    - System information (OS, Python version, etc.) during model saving
```

## Datasets

Datasets used in our research:

1. **Gaofen Image Dataset (GID):** 
   - [Download](https://x-ytong.github.io/project/GID.html)
   - [Reference](https://www.sciencedirect.com/science/article/abs/pii/S0034425719303414/)

2. **Five-Billion-Pixels (FBP) Dataset:** 
   - [Download](https://x-ytong.github.io/project/Five-Billion-Pixels.html)
   - [Reference](https://www.sciencedirect.com/science/article/pii/S0924271622003264/)

3. **Wuhan Urban Semantic Understanding (WUSU) Dataset:**
   - [Download](https://github.com/AngieNikki/openWUSU)
   - [Reference](https://doi.org/10.1080/17538947.2023.2246445)

## Baseline & Comparative Methods
We primarily utilize the [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) library.

The methods we compare include:

- **Global-Local Networks (GLNet):** [Code](https://github.com/VITA-Group/GLNet) | [Paper](https://arxiv.org/abs/1905.06368)
- **CascadePSP:** [Code](https://github.com/hkchengrex/CascadePSP) | [Paper](https://arxiv.org/abs/2005.02551)
- **Progressive Semantic Segmentation (MagNet):** [Code](https://github.com/VinAIResearch/MagNet) | [Paper](https://arxiv.org/abs/2104.03778)
- **Wider Context Transformer (WiCoNet):** [Code](https://github.com/ggsDing/WiCoNet) | [Paper](https://doi.org/10.1109/TGRS.2022.3168697)

## Usage
For step-by-step instructions, please refer to the `Example.ipynb` notebook in this repository.

The agent is trained on images approximately 6000x7000 in size. While it can be tested on images of any size, please note that results may vary.

## Issues and Contributions
Encounter an issue or have a question? Feel free to open an issue in this repository.

## Citation
If you use our Scale Control Agent in your research, please cite our upcoming paper:
- To be announced.

## Additional Resources
Discover more resources and datasets from our group on our [website](http://rsidea.whu.edu.cn/resource_sharing.htm).

Commercial use is prohibited.

For further inquiries, contact me at [liuyinhe@whu.edu.cn](mailto:liuyinhe@whu.edu.cn).

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>