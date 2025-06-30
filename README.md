# FedWSQ: Efficient Federated Learning with WS and DANUQ


This repository is the official implementation of *[ICCV 2025](https://iccv.thecvf.com/)* paper ***["FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization"]()***

By [Seung-Wook Kim](https://www.linkedin.com/in/%EC%8A%B9%EC%9A%B1-%EA%B9%80-003a7310a/), [Seong-yeol Kim](https://github.com/Seongyeol-kim), [Jiah Kim](https://github.com/Kim-Jiah), [Seowon Ji](https://www.linkedin.com/in/seowon-ji-7587741a9/) and [Se-Ho Lee](https://dblp.org/pid/158/9405.html)


<img src="tiny_result.png" alt="teaser" width="1000"/>

> Federated learning (FL) often suffers from performance degradation due to key challenges such as data heterogeneity and communication constraints.
To address these limitations, we present a novel FL framework called FedWSQ, which integrates weight standardization~(WS) and the proposed distribution-aware non-uniform quantization~(DANUQ).
WS enhances FL performance by filtering out biased components in local updates during training, thereby improving the robustness of the model against data heterogeneity and unstable client participation. In addition, DANUQ minimizes quantization errors by leveraging the statistical properties of local model updates. As a result, FedWSQ significantly reduces communication overhead while maintaining superior model accuracy.
Extensive experiments on FL benchmark datasets demonstrate that FedWSQ consistently outperforms existing FL methods across various challenging FL settings, including extreme data heterogeneity and ultra-low-bit communication scenarios. 

## Installation
### ⚙ Dependencies

[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04.4-E95420?logo=Ubuntu&logoColor=white)](https://ubuntu.com/download)
[![Python](https://img.shields.io/badge/Python-3.8.13-3776AB?logo=python&logoColor=white)](https://www.anaconda.com/download)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)

We have exported all required dependencies into `fedwsq.yaml`.  
You can create the environment by running:

```
git clone https://github.com/gymlab/FedWSQ.git
conda env create -f fedwsq.yaml
conda activate fedwsq
```

### 🗂 Dataset
CIFAR-10 and CIFAR-100 will be downloaded automatically. Only Tiny-ImageNet requires manual download.
- [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Tiny-ImageNet](https://www.image-net.org/index.php)

  

### ⚙️ Arguments
The following `arguments` can be adjusted to customize experiments (**default is bold**):

| Argument                       | Options                                                     |
|--------------------------------|-------------------------------------------------------------|
| `--dataset`                    | **`cifar10`** , `cifar100`, `tinyimagenet`                  |
| `--model.wt_bit`               | `1`, `2`, **`4`**, `...`                                    |
| `--split.mode`                 | **`dirichlet`**, `iid`                                      |
| `--split.alpha`                | `0.03`, `0.05`, `0.1`, **`0.3`** , `0.6`, `...`             |
| `--trainer.participation_rate` | `0.02`, **`0.05`**, `...`                                   |
| `--trainer.num_clients`        | **`100`** , `500` , `...`                                   |
| `--quantizer.random_bit`       | **`none`** , `fixed_alloc` , `rand_alloc`                   |

**⚠️ Note** : When `--split.mode` is set to `iid`, `--split.alpha` is ignored.
- To enable **Fixed-Bit Allocation (FBA)**, set `--quantizer.random_bit=fixed_alloc`.
- To enable **Dynamic-Bit Allocation (DBA)**, set `--quantizer.random_bit=rand_alloc`.

 
### 📌 Quick Start
> CIFAR-10, 100 clients, Dirichlet (0.3) split, 5% participation (**default**)  
```
python3 federated_train.py visible_devices="0" client=base server=base dataset=cifar10 trainer.num_clients=100 split.mode=Dirichlet split.alpha=0.3 trainer.participation_rate=0.05 quantizer=WSQLG, quantizer.random_bit: 'none'
```

> CIFAR-10, 100 clients, Dirichlet (0.1) split, 5% participation, FBA
```
python3 federated_train.py visible_devices="0" client=base server=base dataset=cifar10 trainer.num_clients=100 split.mode=dirichlet split.alpha=0.1 trainer.participation_rate=0.05 quantizer=WSQLG quantizer.random_bit: 'fixed_alloc'
```

> CIFAR-100, 500 clients, Dirichlet (0.3) split, 2% participation, DBA
```
python3 federated_train.py visible_devices="0" client=base server=base dataset=cifar100 trainer.num_clients=500 split.mode=dirichlet split.alpha=0.3 trainer.participation_rate=0.02 quantizer=WSQLG quantizer.random_bit: 'rand_alloc'
```

> Tiny-ImageNet, 100 clients, iid, 5% participation
```
python3 federated_train.py visible_devices="0" client=base server=base dataset=tinyimagenet trainer.num_clients=100 split.mode=iid trainer.participation_rate=0.05 quantizer=WSQLG quantizer.random_bit: 'none'
```

## References
### 📚 Citation

If you use this code in a publication, please cite our paper.

```bibtex
@inproceedings{kim2025Efficient,
  author    = {Kim, Seung-Wook and Kim, Seong-Yeol and Kim, Jiah and Ji, Seowon and Lee, Se-Ho},
  title     = {Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization},
  booktitle = {ICCV},
  year      = {2025},
}


```
### 🙏 Acknowledgement

This repository builds upon the excellent framework provided by [FedACG](https://github.com/geehokim/FedACG). Thanks to the original authors for their great contribution.


