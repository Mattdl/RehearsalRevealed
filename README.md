# Rehearsal Revealed: The limits and merits of revisiting samples in continual learning
**[Open-Access paper](https://openaccess.thecvf.com/content/ICCV2021/html/Verwimp_Rehearsal_Revealed_The_Limits_and_Merits_of_Revisiting_Samples_in_ICCV_2021_paper.html)** | **[Open-source MIT license](LICENSE)** | **[Avalanche framework](https://github.com/ContinualAI/avalanche)**

This is the official Pytorch based codebase for the *Rehearsal Revealed* paper, published at ICCV 2021.
We examine replay/rehearsal in continual learning, in the perspective of loss landscapes.
All experiments can be reproduced, from analysing the loss contours, linear interpolations paths and gradients, 
to significantly improving rehearsal with Ridge Aversion.

## Getting started
This code has been tested with
- Python 3.8
- Pytorch 1.7.1
- [Avalanche v0.0.1](https://github.com/ContinualAI/avalanche/tree/v0.0.1)


The requirements for this work are the basic Avalanche dependencies.
You can install them by using the [install script](install_script.sh) or take a look at the script to install the 
[environment](environment.yml) manually.



## Project structure

Explanation for the different experiments can be found in their respective folders.
- [Contour Experiments](./contour_exp): Analyse contours in loss landscape.
- [Linear Path Experiments](./linear_path_exp): Analyse linear interpolation paths in loss landscape.
- [Gradient Norm experiments](./grad_norm_exp): Analyse gradient norms in loss landscape.
- [Ridge Aversion Experiments](./ridge_aversion_exp): Performs extra updates only on rehearsal memory after convergence.

Other modules:
-  [Framework](framework): Root directory of the framework modules
    - [avalanche](framework/avalanche): Avalanche framework for the [Ridge Aversion Experiments](./ridge_aversion_exp).
    - [cole](framework/cole): Continual Learning (cole) framework optimized for loss landscape analysis.
- [data](data): MiniImagenet data folder, containing split. Place preprocessed pickled file here.

**MiniImagenet Benchmark**: First download the 'miniImageNet.pkl' file in `./data` and copy the file `miniimagenet_split
_20.txt` into the same folder. The pickled object should be a dict with keys `data` and `labels`, each with numpy(-like)
arrays as values. The data contains 500 train and 100 test samples per class.
See also the [Mini Imagenet tools](https://github.com/yaoyao-liu/mini-imagenet-tools) for preprocessing.

       
       
## Credits 
- Consider citing our paper upon using this repo:

        @InProceedings{Verwimp_2021_ICCV,
            author    = {Verwimp, Eli and De Lange, Matthias and Tuytelaars, Tinne},
            title     = {Rehearsal Revealed: The Limits and Merits of Revisiting Samples in Continual Learning},
            booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
            month     = {October},
            year      = {2021},
            pages     = {9385-9394}
        }

- Thanks to the [Avalanche framework](https://github.com/ContinualAI/avalanche) v0.0.1 for the Ridge Aversion Experiments.
