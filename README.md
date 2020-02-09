# A Simple Machine Learning Pipeline Using R

This is a simple machine learning pipeline to predict housing prices based on features such as region, distance to MRT, number of shops and date

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What are the things you would need to run the programme?

```R and RStudios. Please note that the necessary packages will be installed and loaded using the R scripts```

## Running the Exploratory Data Analysis Notebook Online

Exploratory data analysis can be carried out through the following link:
```
https://aznoryusof.shinyapps.io/HousingSales_XindianTaipei_ShinyV1/
```
The source code can be found in the document titled "eda.Rmd"

## Running the Machine Learning Pipeline

Run the "MLPipeline.R" script to execute the machine learning task. The script will conduct machine learning on the data, and output the results into four folders within the "mlp output" subdirectory.

The following are the steps taken for building the model and evaluating its performance: 
1. Set up the working directories 
2. Create useful functions 
3. Automatic install and load packages 
4. Read and clean the data
5. Carry out k-means clustering on the selected features representing locations
6. Pre-process the data
7. Deal with missing values
8. Explore the data 
9. Split data into train and test datasets 
10. Set up parameters of the models 
11. Train the models 
12. Evaluate the performance

The following results of the machine learning tasks can be attained in the relevant folders created:
1. Clustering Analysis
2. Exploratory Data Analysis
3. Model Performance, including the Ensembles
4. Evaluation of Model Performances
