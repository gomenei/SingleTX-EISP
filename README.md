# Electromagnetic Inverse Scattering from a Single Transmitter

**CVPR 2026**

This repository contains the PyTorch implementation of the paper **"Electromagnetic Inverse Scattering from a Single Transmitter"**.

<div>
    <a href="https://gomenei.github.io/SingleTX-EISP/"><strong>🌐 Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2506.21349"><strong>📄 Paper</strong></a> |
    <a href="https://youtu.be/HaSAXk7yOPE"><strong>▶️ Video</strong></a>
</div>

## TODO :

- [ ] Release training dataset and train code 
- [ ] Release data processing code


## 1. **Introduction**
<p style="display: inline;">
  We propose a fully end-to-end, data-driven framework for Electromagnetic Inverse Scattering Problem. It uses data priors to overcome the information scarcity inherent in sparse transmitter setups. Our method thereby achieves robust performance even with a single transmitter and enables real-time inference with over 70,000x speed up..
</p>
<div style="text-align: center;">
    <img src="assets/teaser.png" alt="Teaser Figure" width="100%">
</div>

## 2. **Preparation**

### Environment Setup

Create and activate conda environment:

```bash
conda create -n GenEISP python=3.11.10
conda activate GenEISP
pip install -r requirements.txt
```

### Dataset Setup
1. **Download the test datasets**  
   Please download the test datasets from [GoogleDrive](https://drive.google.com/file/d/1Lqp7uqJ5Dix-6tAFecIHwV0d7dVHeIem/view?usp=sharing).

2. **Extract and organize the data**  
   Unzip the downloaded files and place them in the `data/test/` directory. The folder structure should look like this:
   ```
   GenEISP/
   ├── data/
   │   └── test/
   |       ├── 3Dmnist_test/
   │       ├── cylinder/
   │       ├── mnist/
   │       └── IF/
   ```

   - `cylinder/`, `mnist/`, and `IF/` correspond to different test datasets.
   - Each subfolder contains the relevant test samples.

3. **Training data**  
   The training dataset and training code will be released soon. Please stay tuned for updates.

### Pre-trained Model Setup
1. **Download the pre-trained model**  
   Download the pre-trained model weights from [GoogleDrive](https://drive.google.com/file/d/1m_76tn2uw8Sd8NZvGURp4lBDpLoMtdk0/view?usp=sharing).

2. **Organize the model weights**  
   Unzip the downloaded files and place them in the `model/` directory. 


## 3. **Evaluation**

### ⚙️ **3.1. Configuration File (`config.yaml`)**

Configuration files follow the naming convention:  
**`[dataset_name]_noise[level]_N[transmitter_number].yaml`**, where each part indicates the experimental setup:

| Example File                | Components                        | Meaning                                             |
|-----------------------------|-----------------------------------|-----------------------------------------------------|
| `cylinder_noise05_N16.yaml`  | `cylinder` + `noise05` + `N16`     | Cylinder dataset, 5% noise, 16 transmitter           |
| `IF_FDE_noise00_N8.yaml`    | `IF` + `FDE` + `noise00` + `N8`   | Real-world IF dataset, FoamDielExt case, 0% noise, 8 transmitters |
| `IF_FDI_noise00_N8.yaml`    | `IF` + `FDI` + `noise00` + `N8`   | Real-world IF dataset, FoamDielInt case, 0% noise, 8 transmitter |
| `IF_FTD_noise00_N18.yaml`    | `IF` + `FTD` + `noise00` + `N18`   | Real-world IF dataset, FoamTwinDiel case, 0% noise, 18 transmitters
| `mnist_noise30_N16.yaml`     | `mnist` + `noise30` + `N16`        | MNIST dataset, 30% noise, 16 transmitters            |
| `mnist_noise05_N1.yaml`     | `mnist` + `noise30` + `N1`        | MNIST dataset, 30% noise, 1 transmitters            |

**Note:**  
If you want to change the number of transmitters, you can modify `experiment.channels` in the configuration `yaml` file.  
For example:
```yaml
experiment:
  channels: [0, 1, 2, 3]  # Use 4 transmitters, represent channel 0, 1, 2, 3
```
The length of the `channels` list determines the transmitter number.

### ▶️ **3.2. Run Evaluation**
To evaluate the model on your chosen dataset, run one of the following commands:

- **For 2D datasets (e.g., cylinder, mnist, IF):**
```bash
python main.py --config your_path_to_config --mode test
```

- **For 3D datasets (e.g., 3Dmnist):**
```bash
python main.py --config your_path_to_config --mode 3Dtest
```

## 4. Training

Coming Soon.

<!-- ## 📄 Acknowledgments

TODO -->
## Citing
If you find this work useful for your research, please consider citing:

```bibtex
@article{cheng2026electromagneticinversescatteringsingle,
  author    = {Yizhe Cheng and Chunxun Tian and Haoru Wang and Wentao Zhu and Xiaoxuan Ma and Yizhou Wang},
  title     = {Electromagnetic Inverse Scattering from a Single Transmitter},
  journal   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}, 
}
```

