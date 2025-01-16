
This simple project is designed to give new users a very quick overview on how to execute tasks with git commands and to get a simple understanding on how to use branches for development as a beginner.

**Use the raw files in 3 different stages to try having a main feature while adding feature 1 and feature 2 as two separate branches and then do merges as follows:**

Create a file called main.py or similar and copy the code from raw file main.py into it and save.

**After that use the following commands:**

git add main.py
git commit -m "Initial commit: Added DataFrame with European cities"

**Create a new branch and add a the raw file feature 1.py code:**

git checkout -b feature/add-country


**after the code is updated do a commit:**

git add main.py
git commit -m "Feature: Added Country column to DataFrame"

-more to come-