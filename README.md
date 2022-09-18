Detecting emotion from speech is a tricky thing. Research is still going on to figure out what low-level descriptors and high-level descriptors make the ideal pair for detecting emotion. Here I have tried to classify based on MFCC. Train EstimatorTest.py with your data to view results.

For the experiments and results, look at the "Speech Emotion Recogntion.xlsx" file in the docs folder.

This ReadMe file only corresponds to the EstimatorTest.py file. The rest of all the.py files are experiments done by me.

Datsets used

IEMOCAP-https://sail.usc.edu/iemocap/

RAVDESS-https://zenodo.org/record/1188976#.XNPViHUzbCI

EmoDB-http://emodb.bilderbar.info/docu/ (The language is German, but we are not converting it to text.)

SAVEE-http://kahlan.eps.surrey.ac.uk/savee/Introduction.html


Here I calculated the mean signal strength as 32000, which is emperically calculated for the given dataset. If you are using different datasets, this figure will change. Calculate the mean signal strength of your dataset by eliminating outliers.

Point "rootdir" to the IEMOCAP dataset.
Point "_DATA_PATH" to the combined(RAVDESS + EmoDB + SAVEE) dataset.
For all these four datasets (IEMOCAP + RAVDESS + EmoDB + SAVEE), scale down the frequency to 16000Hz and set the channel to Mono. (Librosa library can be used for this.)

Functions which I have not mentioned below are mostly self-explanatory.

The function "get_feature_vector_from_mfcc" extracts the MFCC feature vector from a given wav file. For deciding the shape of the MFCC vector, I tried (13,), (39,) and (40,) these 3 shapes. There is no noticeable difference in accuracy. In this code shape, the MFCC feature vector is 39. I calculated the signal length of each wav file, and if the calculated signal length is less than the mean signal strength, we pad it with zeros; if it's greater, we crop the wav file.

Function "get_data" gets data from all the 4 folders and calls the get_feature_vector_from_mfcc function.

The "attention" function accepts input files, assigns weights, and returns output and alphas.

The function "my_model" labels input features as input. Model architecture is defined here. In this file, I used 2 layers of CNN + BiLSTM + Attention. There are three modes(Train, Predict, and Eval) we don't have to pass explicitly. This function returns a tf-estimatorspec object.

Function "train" takes a tensorflow config as input. The config consists of info about checkpoints, steps needed to be saved, when to display the logging hook and similar information. Here I gave training steps of 10,000 and eval steps of 7. (This number depends on your number of eval files and batch size). Give the path to the folder where you want to save your checkpoint files.

The function "pred_model" takes a file and outputs emotions from the saved model.






