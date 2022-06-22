# anime-characters-recognition

This project aims to recognize every favorate character !!!

The model runs in browser by [onnx runtime web](https://cloudblogs.microsoft.com/opensource/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/).

Now, this model is [MobileNetV3-Large](https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/).

One can train this model by putting custom data folder(named as 'anime-pictures') at '/model/data/'. The folder structure for 'anime-pictures' is assumed to be
 ```
anime-pictures
│
└──train
|    |
|    └──work1
|       |
|       └───character1_1
|          │   image1_1_1.file
|          │   image1_1_2.file
|          │   ...
|   
└──test
```
Then, use train.py to train.
```
python train.py
```
Two files will be generated:

*   characters.json: describe the characters
*   anime.onnx: the model