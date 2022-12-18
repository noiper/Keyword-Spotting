UNI: fs2776

Name: Fengshuo Song

Date: Dec. 18th, 2022

Project Title: Transformer Model for Keyword Spotting Based on MFCC Features
Project Summary: The aim of this project is to build an efficient transformer model based on MFCC features. The model takes in a sequence of MFCC features which are extracted from the utterance of a speaker, and output the keyword corresponding to the utterance. I trained the model using Google Speech Commands Dataset Version 1 and Version 2 on 3 different tasks: 12-class classification, 21-class classification and all-class classification. There are 2 different configurations of MFCC: low resolution (13 dimension) and high resolution (40 dimension). In addition, I also analyzed the model complexity using the number of parameters and number of multiply-accumulate (MAC) operations.

Tools:  1. make\_mfcc: extract the MFCC features
