A deep learning approach for uncertainty quantification based on "A Neural Network Approach for Uncertainty QUantification for Time Dependent Problems with Random Parameters".

1. Make sure your python version is higher than 3.5. The object 'dict' has changed its properties.
2. All the dependencies are listed in "requirements.txt".
3. For the plot part, you have to use "python -m visdom.server" before running the code. 
Enter the link "http://localhost:8097/" to check the plots.
4. Use "python main.py train" to train the model. To change the default parameters in config, just add "--args=xx". 
5. Have not tested on a Nvidia GPU.
