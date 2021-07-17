# TableNet UNet based
UNICAMP - FEEC - IA376I

This repository consists on a Pytorch implementation of [TableNet](https://arxiv.org/abs/2001.01469) using a UNet archtecture instead of classical VGG backend as originally proposed.

Please refer to original code base [OCR TableNet repository](https://github.com/tomassosorio/OCR_tablenet) and [Unet repository](https://github.com/mateuszbuda/brain-segmentation-pytorch) to get the original source codes.


To training or predict, you should first install the requirements by running the following code:

```bash
pip install -r requirements.txt
```

To train is only needed the `train.py` file which can be configured as wanted.
`marmot.py` and `tablenet.py` are inheritance of Pytorch Lighting modules: `LightningDataModule` and `LightningModule`, respectively.

```bash
 python predict.py --model_weights='<weights path>' --image_path='<image path>'
```

or simply:
```bash
 python predict.py
```

To predict with the default image.


