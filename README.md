# XAI for colonoscopy video data classification
This repository contains the official PyTorch implementation of the paper ***Show Me the Polyp! Evaluating KernelSHAP and Grad-CAM Explanations for Colonoscopy Video Analysis with EndoFM-LV*** by Silvie Opolka, Oleksandr Pometun, Krishnendu Bose, Srishti Pandey, Yulia Terekhova, Franziska Tietze, Sarmad Shoaib Baig, and Lukas Niehaus. 

## Getting started
This repository uses Git LFS (Large File Storage). To ensure you can properly clone and work with the repository, please make sure that Git LFS is installed on your system before pulling the project. Without Git LFS, you will only see pointer files instead of the actual data. You can run `git lfs install` in the terminal. You can check if the installation was successful via `git lfs version`. <br>
We suggest using [Anaconda](https://www.anaconda.com/download/success) to set-up and handle your virtual environment. Make sure to add it to your PATH such that you can run commands in your normal terminal. Create a virtual Conda environment via `conda create --name xaiPolyp python=3.10`, activate it via `conda activate xaiPolyp`, and install all requirements via `pip install -r requirements.txt`. 

### Dataset preparation
All datasets can be found in the folder `data`. More information on the datasets can be found in the folder's `README.md` file.

#### PolypGen6
Our dataset _PolypGen6_ contains 6-frame _PolypGen_ subclips and their polyp segmentation masks. Since any public redistribution of the _PolypGen_ dataset is forbidden, you first need to create a personal access token in [Synapse](https://www.synapse.org/). Then copy `.env.example` to `.env`, set `SYNAPSE_AUTH_TOKEN` in `.env` to your Synapse Personal Access Token, and execute `python -m data.prepare_dataset` to download the original _PolypGen_ data and recreate our _PolypGen6_ datasets. <br>
To inspect the `PolypGen6` dataset distribution, run `python -m data.polypGen6.analyze_dataset_distributions`. This replicates the plots in the folder `.data/polypGen6/ds_distribution_info`.

### Model finetuning & evaluation
Everything related to the classification model can be found in the folder `model`. <br>
We finetuned the PolypDiag-trained EndoFM-LV model on _PolypGen6_. To replicate our finetuning, run `python -m model.finetuning --device "cuda" --seed 42 --epochs 6 --batch_size 16 --lr 0.001 --patience 6 --temporal_reverse_prob 0.0 --horizontal_flip_prob 0.0 --vertical_flip_prob 0.0 --brightness_jitter 0.0 --contrast_jitter 0.0 --saturation_jitter 0.0 --hue_jitter 0.0 --gaussian_blur_prob 0.0 --motion_blur_prob 0.0 --rotation_prob 0.0 --random_crop_prob 0.0 --perspective_prob 0.0 --gaussian_noise_prob 0.0`. <br>
To compare the finetuned model with the pretrained model on their performance on the test _PolypGen6_ dataset, run `python -m model.inference --device "cuda"`. <br>
Detailed information on the classification models can be found in the folder's `README.md` file.

### Explanation generation
To be able to generate the explanation with your finetuned model of interest, you must rename the pth file to `polypgen_finetuned_model.pth` and place it in `./model/checkpoints`. <br>
To generate the explanations for the validation dataset and pretrained model, run: `python -m xai_methods.get_explanations --dataset "val" --model "pretrained" --all_test_samples`. <br>
To generate the explanations for the test dataset and pretrained model, run: `python -m xai_methods.get_explanations --dataset "test" --model "pretrained" --all_test_samples`. <br>
To generate the explanations for the validation dataset and finetuned model, run: `python -m xai_methods.get_explanations --dataset "val" --model "finetuned" --all_test_samples`. <br>
To generate the explanations for the test dataset and finetuned model, run: `python -m xai_methods.get_explanations --dataset "test" --model "finetuned" --all_test_samples`. <br>
If you do not have a strong compute, you can decrease the `n_samples` parameter for KernelSHAP in `./xai_methods/apply_KernelSHAP.py` for faster computation. However, be aware that reducing the number of perturbation samples reduces the stability and fidelity of KernelSHAP explanations. <br>
The explanations generated for our study can be extracted by running `tar -xzvf pretrained_polypGen6.tar.gz; tar -xzvf finetuned_polypGen6.tar.gz`.

### Plausibility evaluation
We evaluated explanation plausibility for two dataset-model settings: (1) _PolypGen6_ + pretrained, and (2) _PolypGen6_ + finetuned. <br>
To replicate our evaluation, run `python -m evaluation.run_explanation_plausibility`.


## 🛡️ License
This project is under the _Creative Commons Attribution 4.0 International Public License_. See [LICENSE](LICENSE) for details. <br>
The contents taken from the official EndoFM-LV repository are indicated as such and under the _Apache License 2.0_ license. See [LICENSE_EndoFMLV](model/LICENSE_EndoFMLV.txt) for details. 

## 🙏 Acknowledgement
Our code is based on the [EndoFM-LV model](https://github.com/med-air/EndoFM-LV) and [PolypGen dataset](https://doi.org/10.7303/syn26376615). We thank them for making their model/dataset publically available. 

## 📝 Citation
If this repository contributes to your work, please reference our paper.