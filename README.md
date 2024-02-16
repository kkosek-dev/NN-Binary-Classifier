# Optimizing a Neural Network Model for Binary Classification

## Project Intro
The purpose of this project is to create a tool for a non-profit foundation to assess whether applicants for funding will succeed in their organizational goals based on metadata for the organization. In order to accomplish this goal, a neural network model trained on the preprocessed metadata until reaching a plateau at ~80% predictive accuracy in testing. 

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. The training notebook is located in the forefront of the repository. The script will run find on either Jupyter notebook or Google Collab. 
3. Raw data comes from an API call to this [URL](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv)
4. The models directory contains our optimized neural network model.

### Methods Used
* Deep Learning
* Data Preprocessing
* Binary Classification

### Technologies
* Python
* Pandas, jupyter
* Tensorflow, keras

# Project Overview
## Preprocessing
The raw dataset is 12-dimesional containing a singular target dimension which is a binary column signifying whether a particular applicant was successful (1) or unsuccessful (0). The rest of the dimensions I utilize as features for our model, but in this case most of those features are categorical. Thus, the first step was to preprocess these categorical variables into a usable form. The 'EIN' column functions as an ID for every row in the table, so I dropped this altogether. Next, the categorical values were assesed for how many unique values they have in the dataset. I visualized the features with the most uniqueness in a table to investigate a cutoff value for binning. In particular, 'NAME', 'APPLICATION_TYPE', 'CLASSIFICATION',  'AFFILIATION', and 'USE_CASE' were binned in order to reduce the uniqueness by replacing all of unique values that had less entries in the table than a specified cutoff value with "Other". In another instance, the categorical value "INCOME_AMT" was saved as an object type, but functioned similar to a quanitative variable; thus, I preprocessed this feature into 4 distinct bins that spanned from 0-4 based on the increasing values. The data was then encoded using dummies, standard scaled and pushed to the model. All of the aforementioned preprocessing expanded our data into 612 dimensions, which was utilized to train our neural network.

## Optimization Strategy
Our neural network model has 4 layers with 612, 80, 20 & 1 neuron respectively. Through continual trial & error testing, I found that this combination of neurons yielded the highest accuracy in testing. The model utilizes a rectified linear unit activation function initially and switches to the sigmoid activation function in the 1st and 2nd hidden layers. In compiling our model, I elected to assess loss utilizing binary cross entropy given the objective of our task and experimented with different optimizer functions, eventually deciding on root means squared propagation as the most effective optimizer for the uniqueness of these data. In training, the particular model made minimal progress beyond 60 epochs. Initial efforts at optimization occured at both the preprocessing stage as well as the model design stage. For instance, initially I dropped the 'NAME' feature altogether due to the enormous variability, but in optimization, our model was able to capture more of the variance of the data utilizing this feature. Similarly, during the model design stage, the model was tested many times with the bias of the neurons removed or with kernel & activity regularizers, but none of those options made a significant improvement to the overall accuract of our model. By this logic, I determined that the best model was the simplest model with the highest predictive success rate. All in all, our model achieved a 79.98% accuracy in testing. Although the target level of 75% accurate was achieved, I believe further optimization is still possible, but overall, this would be a strong tool for any non-profit with access to applicant metadata. 

## Continuing Efforts
Given the results of this binary classification task, it is worth discussing the inefficiencies of a loosely defined target value. In other words, the target feature we have in this task has 2 values signifying either an applicant was successful or not. In this model, we have an error 1 in every 5 predictions, thus I would recommend recollecting the data as a continuous variable in order to guage the success of a particular applicant. This regression tool would most likely function veryIll given the existing metadata and could be used strategically to assess the predicted degree of success a particular applicant can be expected to have.
