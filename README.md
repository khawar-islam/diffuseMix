# DiffuseMix : Label-Preserving Data Augmentation with Diffusion Models (CVPR'2024)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://diffusemix.github.io/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2405.14881)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]([https://www.linkedin.com/in/khawarislam/](https://www.youtube.com/watch?v=FcM4wgieDmU))
[![code](https://img.shields.io/badge/-Demo-red)](https://github.com/khawar-islam/diffuseMix)
[![Page Views Count](https://badges.toozhao.com/badges/01HRR1Z1PZQZ9PCVJ7MN2Q67HN/blue.svg)](https://badges.toozhao.com/stats/01HRR1Z1PZQZ9PCVJ7MN2Q67HN "Get your own page views count badge on badges.toozhao.com")


<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="DiffusMix">
</p>

<p align="center">
    <img src="images/diffuseMix_flower102.png" alt="DiffusMix Treasure">
</p>

---

### 📢 Latest Updates
- **Mar-11-24**: DiffuseMix paper is released [arxiv link](https://arxiv.org/abs/2405.14881). 🔥🔥

### 🚀 Getting Started
Setup anaconda environment using `environment.yml` file.

```
conda env create --name DiffuseMix --file=environment.yml
conda remove -n DiffuseMix --all # In case environment installation faileds
```

### 📝 List of Prompts 
Below is the list of prompts, if your accuracy is low then you can use all prompts to increase the performance. Remember that each prompt takes a time to generate images, so the best way is to start from two prompts then increase the number of prompts.

```
prompts = ["Autumn", "snowy", "watercolor art","sunset", "rainbow", "aurora",
               "mosaic", "ukiyo-e", "a sketch with crayon"]
```

### 📁 Dataset Structure
```
train
 └─── class 1
          └───── n04355338_22023.jpg
 └─── class 2
          └───── n03786901_5410.jpg
 └─── ...
```
### ✨ DiffuseMix Augmentation
To introduce the structural complexity, you can download fractal image dataset from here [Fractal Dataset](https://drive.google.com/drive/folders/1uxK7JaO1NaJxaAGViQa1bZfX6ZzNMzx2?usp=sharing)
```
`python3 main.py --train_dir PATH --fractal_dir PATH --prompts "sunset,Autumn"
```

### 💬 Citation
If you find our work useful in your research please consider citing our paper:
```
@article{diffuseMix2024,
  title={DIFFUSEMIX: Label-Preserving Data Augmentation with Diffusion Models},
  author={Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### ❤️ Acknowledgment
I am grateful to Adversarial-AutoMixup (@JinXins) for providing the source and target images, which significantly saved me a lot of time. Thank you once again. I am also exceptionally thankful to the author of IPMix, (@hzlsaber), for presenting their method's figures clearly, which greatly aided my paper. Additionally, their timely responses to my concerns saved me considerable time. Lastly, my thanks again go to the author of GuidedMixup, (@3neutronstar), for their insights on datasets and method outputs.
