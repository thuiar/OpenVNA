# Robust-MSA

This is the source code of paper "OpenVNA: An Open-source Framework for Video Noise Analysis".
The Demonstration Video can be viewed on [Youtube](https://youtu.be/NfaG5pr5k7o)

## Deployment

This demonstration system is developed based on the B/S architecture.

The front-end source code is located in the `vue3` folder, while the published web pages can be found in the `vue3/dist` folder. They can be deployed using server softwares such as Nginx. There are two urls need to be adjusted in `dist/config.js` according to the IP address of the server.

The back-end is coded in Python, the package dependencies are listed in `requirements.txt`. Note that the [MMSA-FET](https://github.com/thuiar/MMSA-FET) package needs additional post-installation setups to function properly as documented [here](https://github.com/thuiar/MMSA-FET/wiki/Dependency-Installation). Server ports can be adjusted in `config.py`.

## Citation

If you find this work helpful, please cite us:

```
@article{mao2022robust,
  title={Robust-MSA: Understanding the Impact of Modality Noise on Multimodal Sentiment Analysis},
  author={Mao, Huisheng and Zhang, Baozheng and Xu, Hua and Yuan, Ziqi and Liu, Yihe},
  journal={arXiv preprint arXiv:2211.13484},
  year={2022}
}

@article{yuan2023noise,
  title={Noise Imitation Based Adversarial Training for Robust Multimodal Sentiment Analysis},
  author={Yuan, Ziqi and Liu, Yihe and Xu, Hua and Gao, Kai},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
