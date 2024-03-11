# DiffuseMix : Label-Preserving Data Augmentation with Diffusion Models (CVPR'2024)
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Khawar Islam](https://www.linkedin.com/in/khawarislam/)\*, [Muhammad Zaigham Zaheer](https://www.linkedin.com/in/zaighamzaheer//)\*, [Arif Mahmood](https://www.linkedin.com/in/arif-mahmood-36875ab1/), and [Karthik Nandakumar](https://www.linkedin.com/in/karthik-nandakumar-5504465/)

#### **FloppyDisk.AI, Mohamed bin Zayed University of Artificial Intelligence, Information Technology University**

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://www.linkedin.com/in/khawarislam/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.linkedin.com/in/khawarislam/)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://www.linkedin.com/in/khawarislam/)
[![demo](https://img.shields.io/badge/-Demo-red)](https://www.linkedin.com/in/khawarislam/)

---

## 📢 Latest Updates
- **Mar-11-24**: Extending version will be available on IEEE PAMI
- **Nov-28-23**: DiffuseMix paper is released [arxiv link](https://arxiv.org/abs/2311.15826). 🔥🔥
- 📦 Code, models, and datasets coming soon! 🚀
---

## Getting Started
Setup anaconda environment using `environment.yml` file.

`conda env create --name DiffuseMix --file=environment.yml`

`conda remove -n DiffuseMix --all` # In case environment installation failed

## List of Prompts 
Below is the list of prompts, if your accuracy is low then you can use all prompts to increase the performance. Remember that each prompt takes a time to generate images, so the best way is to start from two prompts then increase the number of prompts.

`prompts = ["Autumn", "snowy", "watercolor art","sunset", "rainbow", "aurora",
               "mosaic", "ukiyo-e", "a sketch with crayon"]`


## Dataset Structure
`train
 └─── class 1
          └───── n04355338_22023.jpg
 └─── class 2
          └───── n03786901_5410.jpg
 └─── ...`

`python3 main.py --train_dir PATH --fractal_dir PATH --prompts sunset,Autumn
`

