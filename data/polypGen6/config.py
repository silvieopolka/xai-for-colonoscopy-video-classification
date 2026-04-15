"""The PolypGen6 dataset configuration object."""

class CfgPolypGen6:
    class DATA:
        TRAIN_CROP_SIZE = 224   # this stayed the same across backbone, pretrained, and finetuned models
        NUM_FRAMES = 6          # pretrained model is optimized for 16-frame inputs, and finetuned model for 6-frame inputs

    class MODEL:
        NUM_CLASSES = 2         # this stayed the same across pretrained and finetuned models

    class TIMESFORMER:
        ATTENTION_TYPE = "divided_space_time"
        BACKBONE_MODEL = "./model/checkpoints/endofm_lv.pth"                    # provided by https://github.com/med-air/EndoFM-LV.git
        PRETRAINED_MODEL = "./model/checkpoints/polypdiag_finetuned_model.pth"  # provided by https://github.com/med-air/EndoFM-LV.git
        FINETUNED_MODEL = "./model/checkpoints/polypgen_finetuned_model.pth"    # finetuned by us on PolypGen6