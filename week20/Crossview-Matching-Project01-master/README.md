###### # Ground-to-Aerial Image Geo-Localization Experiments
###### # Siam-FCANet18 and Siam-FCANet34
###### # Files for experiments on CVUSA dataset
###### # Python 2.7 (3.6 and 3.7 are also working) and Pytorch 0.4.0 or later versions
###### # The CVUSA dataset can be accessed from this link: https://github.com/viibridges/crossnet
###### # The trained model weights for initialization and test can be accessed from this link: https://pan.baidu.com/s/1WBbxhBhHX7cRo3J9RFUhtg password to extract: 2xvc (note: I am sorry that my dropbox meets a problem of file sharing, I may update the dropbox link after it is working)

###### # For training on CVUSA, Please directly run "train_CVUSA_01.py" with corrected paths to dataset and the model. Models trained by VH dataset are recommended to be used as initialization. 
###### # For evaluation, two ways are both working: (1) run "train_CVUSA_01.py" with setting (epoch>-1) in the "### ranking test" sub-section, (2) run "Feature_vectors_generation.py" to get the features and then run "RankingTest.py" or "RankingTest_ForBigMat.py" to output the results at corresponding metrics (the parameter "length" is used to control the metrics, i.e., recall at top-k or top k%)
###### # Please note that: when applying HER-loss to train un-normalized models (i.e., embedding features are not normalized), sometimes the trainig process may crash with "NaN" occurs at the very beginning, because values of some variables in the loss function are getting too big under the current initializations of some learning layers, or the learning rate is too big. If this situation happens, please re-run the "train_CVUSA_01.py" (the better with a smaller learning rate).

###### #Evaluation on CVUSA 
###### Recall@ Top 1%: 98.491% (Net18), 98.345% (Net34 old model)
###### Recall@ Top-1:  51.013% (Net18), 42.863% (Net34 old model)
