# CycleGAN-pytorch  
the pytorch version of pix2pix 

## Requirments 
- CUDA 8.0+  
- pytorch 0.3.1    
- torchvision  

## Datasets 
- Download a cycleGAN dataset (e.g.maps):  
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
``` 
## Train a model:  
```   
python cycleGAN.py --data_root 'your data directory'   
```   
## Result examples  
  
### epoch-199   
![image](https://github.com/TeeyoHuang/CycleGAN-pytorch/blob/master/result/218800-199.png) 

From **top** to **bottom**：
A-->fake_B-->recon_A + ident_B;  B-->fake_A-->recon_B + ident_A
   
## Reference    
[1][Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)  
```   
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}   
```   

## Personal-Blog 
[teeyohuang](https://blog.csdn.net/Teeyohuang/article/details/82729047)
