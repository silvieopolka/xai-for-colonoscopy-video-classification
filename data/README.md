This folder contains everything related to the **_PolypGen_ dataset**. <br>

Dataset DOI: https://doi.org/10.7303/syn26376615 <br>
ID of the folder to be downloaded: syn45200214

## Data usage rules
Any kind of redistribution of the _PolypGen_ dataset or parts of it is not allowed, which includes our _PolypGen6_ dataset. <br>
When using this dataset, the following references need to be made: <br>
[1] Ali, S., Jha, D., Ghatwary, N. et al. A multi-centre polyp detection and segmentation dataset for generalisability assessment. Sci Data 10, 75 (2023). https://doi.org/10.1038/s41597-023-01981-y <br>
[2] Ali, S., Ghatwary, N., Jha, D. et al. Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge. Sci Rep 14, 2032 (2024). https://doi.org/10.1038/s41598-024-52063-x <br>
[3] Ali S, Dmitrieva M, Ghatwary N, Bano S, Polat G, Temizel A, et al. Deep learning for detection and segmentation of artefact and disease instances in gastrointestinal endoscopy. Medical Image Analysis. 2021:102002. https://doi.org/10.1016/j.media.2021.102002 

## What did we modify?
- We only kept the folders containing the **sequence data** for both positive (polyp) and negative (healthy) classes.
- We only kept the **`.jpg` files** of the original frames and segmentation masks.
- We considered the **original labeling** but be aware of the following special cases:
    - There are two videos (`seq1` and `seq7`) inside the positive class folder, that are actually healthy cases but just include some artifacts (light reflections, green patch, remnant stool). We consider them accordingly as negative class instances.
    - We extracted two "healthy" subclips (`seq17_clip1` and `seq17_clip2`) of `seq17`, where the recorded colon does contain polyps - just not in the selected frames.
- The sequence data differs in resolution, with much more representative samples for 1920x1080 px. We only considered data in **two similar resolutions (1920x1080 and 1280x720)**.

| Resolution  | # of Sequences (Polyp) | # of Sequences (Healthy) |
|-------------|------------------------|--------------------------|
| 1920 x 1080 | 7                      | 19 (2)                   |
| 1280 x 720  | 8                      | 0                        |
| 720 x 576   | 1                      | 4                        |
| 1440 x 1064 | 3                      | 0                        |
| 1280 x 1024 | 1                      | 0                        |
| 1704 x 1072 | 1                      | 0                        |

Note: The (2) refers to the samples `seq1` and `seq7`. <br>
Note: We consider the sample in 1280 x 1024 px resolution as a strong outlier, because the circular field of view differs strongly from the other videos. <br>
Note: We consider the samples in 720 x 576 px resolution as strong outliers, because they contain text strongly overlapping the circular field of view. <br>
Note: We consider the samples in 1440 x 1064 px resolution as weak outliers, because they contain text weakly overlapping the circular field of view. <br>
Note: The videos of resolution 1280 x 720 and 1704 x 1072 are quite similar to resolution 1920 x 1080 px in their circular field of view. <br>

- The video frames were collected from six different hospitals with varying conditions. We only considered data from **C4 (Oslo University Hospital) and C6 (University of Alexandria)**. 

- The recording systems varied across hospitals and within the hospital. While the recording system used for each data sample is not documented in the data, we can see notible differences in the resolution and colonoscopic video positioning. We **defined cropboxes** for each datasample (to crop the original frame to only the endoscopic video) such that the resulting data samples become more similar despite different recording systems. We **resized each frame to 224x224 pixels**, since this is the input shape that we use for the EndoFM-LV model. Despite cropping and resizing, the samples of different camera systems might still exihibit differences (e.g. noise structures might be more prominint in originally higher resolution images; the size and shape of the black triangles at each corner can vary slightly).

- For _PolypGen6_, we manually selected 6-frame clips from the videos and divided them into training, validation, and test datasets. A detailed overview on the clips can be found in `meta_data.csv`. Once the datasets have been extracted, the input clips can be found in the `data` folder and the segmentation masks can be found in the `mask` folder. The division into the datasets was based on ...
    - class balance within each dataset
    - patient balance, especially within the validation dataset (identical number of subclips per original clip)
    - avoiding data leakage via a patient-wise split, meaning all subclips from the same colon (i.e, original video) are strictly confined to either the training, validation, or test set
    - an equal number of subclips for validation and test datasets
    - a similar polyp size distribution across datasets

- We added black segmentation masks for the negative cases.