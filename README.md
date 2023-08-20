# xtream AI Assignment

## Introduction

If you read this file, you have passed our initial screening. Well done! :clap: :clap: :clap:

:rocket: The next step to join the AI team of [xtream](https://xtreamers.io) is this assignment. 
You will find several datasets: please choose **only one**.
For each dataset, we propose several challenges. You **do not need to
complete all of them**, but rather only the ones you feel comfortable about or the ones that interest you. 

:sparkles: Choose what really makes you shine!

:watch: The deadline for submission is **10 days** after you are provided with the link to this repository, 
so that you can move at your own pace.

:heavy_exclamation_mark: **Important**: you might feel the tasks are too broad, or the requirements are not
fully elicited. **This is done on purpose**: we wish to let you take your own way in 
extracting value from the data and in developing your own solutions.

### Deliverables

Please fork this repository and work on it as if you were taking on a real-world project. 
On the deadline, we will check out your work.

:heavy_exclamation_mark: **Important**: At the end of this README, you will find a blank "How to run" section. 
Please write there instructions on how to run your code.

### Evaluation

Your work will be assessed according to several criteria, for instance:

* Work Method
* Understanding of the business problem
* Understanding of the data
* Correctness, completeness, and clarity of the results
* Correct use of the tools (git workflow, use of Python libraries, etc.)
* Quality of the codebase
* Documentation

:heavy_exclamation_mark: **Important**: this is not a Kaggle competition, we do not care about model performance.
No need to get the best possible model: focus on showing your method and why you would be able to get there,
given enough time and support.

---

### Employee Churn

**Problem type**: classification

**Dataset description**: [Employee churn readme](./datasets/employee-churn/README.md)

You have just been contracted by Pear Inc, a multinational company worried about its poor talent retention.
In the past few months, they collected data about their new employees. All of them come from classes 
the company is sponsoring, yet many enter Pear just to leave a few months later.
This is a huge waste of time and money.

The HR department of the company wants you to understand what is going on and to prevent further bleeding.

The main sponsor of the project is Gabriele, Head of Talent at Pear.

#### Challenge 1

Pear Inc wants you to understand what are the main traits separating the loyal employees from the others.
**Create a Jupyter notebook to answer their query.**
Gabriele is not an AI expert, so be sure to explain your results in a clear and simple way.
However, you are also told that Fabio, an ML Engineer, will review your work: be sure to provide enough details to be useful for him.

#### Challenge 2

Then, a predicting model.
**You are asked to create a model to predict whether a new employee would churn**.
Gabriele tells you that he would like to know the probability of churn for each employee, so that he could take 
corrective actions.
Fabio has now joined Pear, and has some advice for you: Gabriele does not believe in black-box models, so
be sure to provide him with compelling evidence that your model works.

#### Challenge 3

Wow, the model works great, but why does it? 
**Try and make the model interpretable**, by highlighting the most important features and how each prediction is made.
You'll need to explain your work to both Gabriele and Fabio, so be sure to include clear and simple text, 
but feel free to use advanced techniques, if you feel that it is necessary.

#### Challenge 4

Now, production trial. 
**Develop and end-to-end pipeline to train a model given a new dataset.**
You can assume that the new dataset has exactly the same structure as the provided one: 
possible structural changes will be managed by your fellow data engineers.
Fabio is a clean code lover: make sure not to disappoint him!

#### Challenge 5

Finally, Pear Inc is happy with your results!
Now they want to embed your model in a web application. 
**Develop a REST API to expose the model predictions**.
Again, this is no longer about Gabriele, but Fabio will review and evolve your work.
Be sure to provide him with clean and well-structured code.

---

## How to run
There are 4 different coding files.

The first one is the jupyter notebook. Here I correctly imported the dataset and analyzed it. Afterwards, I developed a Multi-Layer-Perceptron Classifier and tested it. 
Then, I tried to make the model interpretable by the analysis of the most important features of the model and I checked the performances of the model by looking at the main metrics.

File pipeline.py contains the end-to-end pipeline to train a model given a new similar dataset. At the end, the model is saved in order to be recalled by the REST API and save time beacuse the model is not trained again.

File app.py is the REST API which expose the model predictions. It has to run when the client will test the model developed.

File apiclient.py is an example of python file which has to be launched in order to obtain predictions for a test dataset.
