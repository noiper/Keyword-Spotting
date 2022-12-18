a. UNI: fs2776
   Name: Fengshuo Song

b. Date: Dec. 18th, 2022

c. Project Title: Transformer Model for Keyword Spotting Based on MFCC Features

d. Project Summary: The aim of this project is to build an efficient transformer model based on MFCC features. The model takes in a sequence of MFCC features which are extracted from the utterance of a speaker, and output the keyword corresponding to the utterance. I trained the model using Google Speech Commands Dataset Version 1 and Version 2 on 3 different tasks: 12-class classification, 21-class classification and all-class classification. There are 2 different configurations of MFCC: low resolution (13 dimension) and high resolution (40 dimension). In addition, I also analyzed the model complexity using the number of parameters and number of multiply-accumulate (MAC) operations.

e. Tools:
   1. make\_mfcc: extract the MFCC features
   2. kaldiio: convert kaldi object (.ark files) to files that Python can handle. https://github.com/nttcslab-sp/kaldiio
   3. pytorch: machine learning framwork that is to build the transformer model.
 
f. All the files that may be used to test the code are in the directory: /home/fs2776/kaldi/egs/keyword_spotting/kws/.

g. The main script that runs everything is run.sh in /home/fs2776/kaldi/egs/keyword_spotting/kws/. This file includes downloading data, preparing data, feature extraction, and building and decoding the model.
   To run the scripts, first go to the directory /home/fs2776/kaldi/egs/keyword_spotting/ and activate the virtual environment.
   >>> cd /home/fs2776/kaldi/egs/keyword_spotting
   >>> source kws_venv/bin/activate
   After running the commands, there should be (kws_venv) on the left.
   Then go the directory /home/fs2776/kaldi/egs/keyword_spotting/kws and run the main script.
   >>> cd kws
   >>> ./run.sh --stage 3
   The output is shown in the terminal. It should be something like:
     [1/12] Running 12-class classification on dataset v1 using low resolution MFCC features...
     Test accuracy:  0.9248046875
     [2/12] Running 12-class classification on dataset v2 using low resolution MFCC features...
     Test accuracy:  0.9482421875
     ...
     [11/12] Running all-class classification on dataset v1 using high resolution MFCC features...
     Test accuracy:  0.8974609375
     [12/12] Running all-class classification on dataset v2 using high resolution MFCC features...
     Test accuracy:  0.90234375
   There are a total of 12 models with different configurations and I showed the test accuracy for each model.

h. The dataset is Google Speech Command Dataset (https://www.tensorflow.org/datasets/catalog/speech_commands). There are 2 versions, and can be download in http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz and http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz.
   I have already uploaded the datasets and pre-computed MFCC features, so that the decoding won't take too long. To re-compute the MFCC features, run all stages of the main script.
   >>> ./run.sh
   This will take around 1.5 hours to complete on the vm. However, it takes only 5 minutes on my local computer.
   All packages are already installed in the virtual environment. However, if there are some missing packages, run the following commands:
   >>> cd /home/fs2776/kaldi/egs/keyword_spotting
   >>> python3 -m pip install -r requirements.txt
   Also if there are some permission issues (bash:<filename>: Permission denied), run this command:
   >>> chomod +x <filename>
        
