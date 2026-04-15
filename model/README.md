This folder includes code adapted from the [EndoFM‑LV repository](https://github.com/med-air/EndoFM-LV.git), which is licensed under the Apache License 2.0.
The original Apache 2.0 license is included in `LICENSE_EndoFMLV.txt`. <br>

The **pretrained EndoFM-LV model** pth file was downloaded from the official EndoFM-LV repository, and stored in the file `endofm_lv.pth`. All keys, except 'student' and 'teacher' were removed. <br>

The **PolypDiag-finetuned EndoFM-LV model** pth was downloaded from the official EndoFM-LV repository, and stored in the file `polypdiag_finetuned_model.pth`. <br>
For finetuning, the classification head was replaced by a linear layer with two output units. Both, the backbone and classification head parameters were enabled for training. The subclips consisted of 32 frames each.<br>

The `polypdiag_finetuned_model.pth` file contains the checkpoint after finetuning the pretrained EndoFM-LV model on the PolypDiag dataset. <br>
It contains various information on the finetuning, which is categorized into several keys:
- *'epoch'*: They finetuned for 20 epochs.
- *'optimizer'*: They used the AdamW optimizer. 
- *'scheduler'*: They used the Consine Annealing Learning Rate scheduler.
- *'best_f1'*: The best validation F1-score they achieved during finetuning was 0.9491525423728815.
- *'backbone_state_dict'*: The learnable parameters of the backbone, i.e., the pretrained EndoFM-LV model.
- *'state_dict'*: The learnable parameters of the classification head added for finetuning.

They used the following optimizer setting: <br>
```python
optimizer = AdamW(lr=1.5625e-05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
```
<br>
As indicated in the respective files, some files contain code from the [GitHub EndoFM-LV repository](https://github.com/med-air/EndoFM-LV.git). <br>

In our study, we further finetuned the PolypDiag-finetuned EndoFM-LV model (we call it our **'pretrained' model**) on *PolypGen6* data. The resulting *best* model (we call it our **'finetuned' model**) is stored in the file `polypgen_finetuned_model.py`. <br>
We loaded the PolypDiag-finetuned parameters for the backbone and added a new classification head of identical dimension as the classification head used for the PolypDiag dataset. We froze the entire backbone and only trained the classification head parameters on 6-frame _PolypGen_ clips. We applied no data augmentation, because adding data augmentation did not improve the model performance. We finetuned for 6 epochs using a batch size of 16 and a learning rate of 0.001. The model state with the best F1 score on the validation data was chosen (epoch 6). We used CUDA Toolkit 11.5 (nvcc V11.5.119) for GPU compilation. <br>

Our finetuned model performs better than the pretrained model. <br>
- [PRETRAINED ENDOFM-LV] Accuracy: 0.44999998807907104, F1 Score: 0.6206896551724138, Precision: 0.47368421052631576, Recall: 0.9
- [FINETUNED ENDOFM-LV]  Accuracy: 0.75, F1 Score: 0.761904761904762, Precision: 0.7272727272727273, Recall: 0.8