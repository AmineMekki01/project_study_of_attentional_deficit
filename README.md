# 1. project study of attentional deficit
Study of attentional deficit using EEG signals.

This is a school project for the course of Human and machine at Ecole des Mines.

The objective is to classify the epochs according to the class Hit or NoHit. 

I decided to make it in a modular way. But as a deliverable, I will provide a notebook with the whole process as well.

I added a small web app to train (It doesn't take too much time) and test the model.

I added some plots and tables to visualize the data and the results. But i will add full detailed analysis in the notebook. And maybe in the web app as well.

# 2. How to run the web app : There are 3 ways to test the code 

- First download the archive file and extract it. 
- Or clone the repo using : 
- ```git clone https://github.com/AmineMekki01/project_study_of_attentional_deficit.git```
- Install the requirements using : ```pip install -r requirements.txt```

## 1.1. Using the web app :
- If you want to use the web app, you can run the app.py file using : ```python app.py``` in the terminal of the project directory. And it will open the web app in your browser. You can start interacting with it.
- The url is most likely : http://127.0.0.1:5000


## 1.2. Using the notebook :
- If you want to use the notebook, you can use the jupyter notebook in the notebooks folder. And run the cells.


## 1.3. Using the script :
- If you want to run the project as a script, you can use the main.py file. And run it using : ```python main.py``` in the terminal of the project directory. 


# 2. The project structure
- `artifacts` : Where the all project data is stored, patient data, machine learning models, features selection models and scores of testing and training
  - `data` : Patient data 
    - `raw` 
      - `AEP` 
  - `feature_selectors` : Feature selection models
  - `models` : Machine learning models
  - `plots` : Plots of the data
    - `test` : Plots of the testing data
    - `train` : Plots of the training data
  - `scores` : Scores of the testing and training data
    - `testing` : Scores of the testing data
    - `training` : Scores of the training data
  - `uploaded_data` : Data uploaded by the user using the browser.
- `notebooks` : Jupyter notebooks of the project 
    - `solution.ipynb` : the solution notebook of the project.
- `src` : Source code of the project
    - `components` : Components of the project
    - `models` : Machine learning models
    - `pipeline` : Pipeline of the project
    - `utils` : Utils of the project
- `static` : Static files of the web app (css, js)
- `templates` : Templates of the web app (html)
- `main.py` : Main file of the project if you want to run the project as a script
- `app.py` : Flask app that runs the web app
- `requirements.txt` : Requirements of the project to install using : pip install -r requirements.txt