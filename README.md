# sdk-tutorial

This is a short tutorial for using the [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

First of all you need to download the SDK from [here](https://github.com/A2Zadeh/CMU-MultimodalSDK).
After that you will need to set up some file paths in `./constants/paths.py`. The comments should be self-explanatory.

If you have pretrained embeddings such as [GloVe](https://nlp.stanford.edu/projects/glove/), you can setup the path to them in `./constants/paths.py` for improved performance.

Note that in the `./data/` directory there is a `./CMU_MOSI_ModifiedTimestampedWords.csd`. This is an updated copy of transcripts compared to what the [SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) is currently offering and improves the performance by a observable margin. If you are changing the `DATA_PATH` in `./constants/paths.py`, please also copy this file to your specified location.

Then you are good to go, just run the `./tutorial_interactive.ipynb` and walk through its content!
