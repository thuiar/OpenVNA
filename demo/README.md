# Robust-MSA

This is the source code of paper "OpenVNA: An Open-source Framework for Video Noise Analysis".
The Demonstration Video can be viewed on [Youtube](https://youtu.be/NfaG5pr5k7o)

## Deployment

This demonstration system is developed based on the B/S architecture.

The front-end source code is located in the `vue3` folder, while the published web pages can be found in the `vue3/dist` folder. They can be deployed using server softwares such as Nginx. There are two urls need to be adjusted in `dist/config.js` according to the IP address of the server.

The back-end is coded in Python, the package dependencies are listed in `requirements.txt`. Note that the [MMSA-FET](https://github.com/thuiar/MMSA-FET) package needs additional post-installation setups to function properly as documented [here](https://github.com/thuiar/MMSA-FET/wiki/Dependency-Installation). Server ports can be adjusted in `config.py`.

### Back-end Setups

1. Download pretrained models ([BERT](https://huggingface.co/google-bert/bert-base-uncased) for text embedding and [Wav2Vec](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) for ASR and Alignment), and modify the corresponding settings in `demo/config.py` file.

2. Setup your web server and media server ports in `demo/config.py` file, make sure that no port conflicts have occurred.

3. Evaluating the provided models or user specific multimodal understanding models.

    a. The author provide several common robust multimodal understanding models for evaluation, for reproducing users should download the model checkpoints from [BaiduYunDisk](https://pan.baidu.com/s/1Dq996T--wDZA9d9x2EHoZA?pwd=neij) `提取码: neij` (3.6G in total) and modified the corresponding `PRETRAINED_MODEL` settings in `demo/config.py`

    b. The platform support users to integral other multimodal understanding models into the framework, and preform robustness analysis. Please transfer the corresponding model hyperparameters settings in `demo/model_api/configs` and corresponding model structure in `demo/model_api/models` and corresponding training process (evaluation process) in `demo/model_api/trainers` and modify the `demo/model_api/run.py` to include the specific model in the `ModelApi`.  


### Front-end Setups

1. specific the base_url and static_url config in `demo/vue3/public/config.js`. 

2. install packages and run.
```
$ npm install
$ npm run build     # build dist files
$ npm run serve     # run dev server
```


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
