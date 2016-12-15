# Training Inception on a Custom Dataset with TF-Slim

I forked the whole repo, but all the relevant code is in the `slim_on_customdata` directory (which is itself just a copy of the slim directory). 

1. Follow [these instructions](https://github.com/tensorflow/models/tree/master/inception#how-to-construct-a-new-dataset-for-retraining) to create a custom set of TFRecords.

2. In `slim_on_customdata/datasets` edit the testset.py script.
  - `_FILE_PATTERN` needs to match the format of the TFRecord names of your custom data
  - `SPLITS_TO_SIZES` needs to match number of training and validation images
  - `NUM_CLASSES` needs to match the number of classes/labels you're training on
  - OPTIONALLY, rename testset.py to match the name of your custom dataset

2b. If you renamed testset.py, edit dataset_factory.py (also in the datasets directory), and change the testset entry in `datasets_map` to match your chosen dataset name.

3. The training is kicked off by the `train_inception_on_testset.sh` script, which will call `train_image_classifier.py` which all the necessary parameters. 
  - point `TRAIN_DIR` and `DATASET_DIR` to the appropriate directories
  - IF YOU RENAMED TESTSET, change the `--dataste_name` parameter to match
  - Set the rest of your parameters as you will. The ones here aren't necessarily recommended, I chose them randomly when I started. 
  - If you want to both train and evaluate, de-comment the evaluation section in the script.

These are roughly the steps to get something to start training without getting caught up by errors. No guarantees it'll train a useable model.







## TensorFlow Models

This repository contains machine learning models implemented in
[TensorFlow](https://tensorflow.org). The models are maintained by their
respective authors.

To propose a model for inclusion please submit a pull request.


### Models
- [autoencoder](autoencoder) -- various autoencoders
- [inception](inception) -- deep convolutional networks for computer vision
- [namignizer](namignizer) -- recognize and generate names
- [neural_gpu](neural_gpu) -- highly parallel neural computer
- [privacy](privacy) -- privacy-preserving student models from multiple teachers
- [resnet](resnet) -- deep and wide residual networks
- [slim](slim) -- image classification models in TF-Slim
- [swivel](swivel) -- the Swivel algorithm for generating word embeddings
- [syntaxnet](syntaxnet) -- neural models of natural language syntax
- [textsum](textsum) -- sequence-to-sequence with attention model for text summarization.
- [transformer](transformer) -- spatial transformer network, which allows the spatial manipulation of data within the network
- [im2txt](im2txt) -- image-to-text neural network for image captioning.
