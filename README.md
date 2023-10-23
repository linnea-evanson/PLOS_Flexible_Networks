# PLOS_Flexible_Networks

#### Code to recreate the paper "Biomimetic computations improve neural network robustness", Linnea Evanson <sup>1</sup>, Maksim Lavrov <sup>1</sup>, Iakov Kharitonov <sup>1</sup>, Sihao Lu, Andriy S. Kozlov
<sup>1</sup> Equal contribution

* The code for implementing the flexible layer is found in network_definitions.py.
* To recreate main results, run train_VGG16.ipynb, import the network of interest from network_definitions.py. This script saves validation accuracies. Uncomment the line to save model weights if you wish to run further analyses. 
* Once you have trained a VGG16 model, run adverserial_FGSM.ipynb (to recreate Fig. 4D) or adverserial_PGD.ipynb to recreate the adverserial attack results.
* Train a network on a downsampled subset of data using this script: train_downsampled.ipynb. To recreate our results run for seeds 1-10. 
* The spectral analysis figures are created with test_robustness.ipynb, out_of_distribution_inference.ipybn, or bandpass_inference.ipynb. 

