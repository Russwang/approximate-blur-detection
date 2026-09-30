# Data setup

The paper uses selected clean, defocus-blurred, and motion-blurred images from a Kaggle blur dataset. Obtain the data from the original providers and check their terms:

- [Kaggle blur dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset)
- [Dataset project](https://github.com/Kwentar/blur_dataset)

The original notebooks expect this layout, relative to the repository root:

```text
data/
  clean/       # *_S.JPG: clean references
  defocus/     # *_F.JPG: corresponding defocus images
  mobile/      # *_M.JPG: corresponding motion images (historical folder name)
```

Files with a shared base identifier should refer to the same scene. The notebook loops associate clean and blurred images by traversal order, so verify the order or change the pairing to use file identifiers before conducting a new evaluation. The command-line demo takes an explicit reference/query pair instead.

The research notebooks crop rows 128:1152 and columns 256:1792, then resize by 0.5. Use source images large enough for that crop. These settings yield a 512 x 768 processed image for a full crop; they differ from the paper's 2048 x 1280 complexity example.

Dataset files and generated outputs are ignored by Git. The default command-line example generates its own synthetic inputs and needs no external image downloads.
