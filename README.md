# pyHEXtraction

## Prerequisites after Cloning the repository:

1. Create Python Virtual Environment:
    ```
    python -m venv env
    ```
2. Activate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/activate
    ```
    **Windows**
    ```
    env\Scripts\activate
    ```
3. Load Required libraries
    ```
    pip install -r requirements.txt
    ```
4. Give Script permissions in Linux:<br>
    ```
    chmod +x main.py
    ```
5. Deactivate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/deactivate
    ```
    **Windows**
    ```
    env\Scripts\deactivate
    ```
## Save new libraries installed in the requirements.txt
Inside the env:
```
python -m pip freeze > requirements.txt
```

## Normal Workflow to launch scripts:
1. Activate Virtual Environment:<br>
    **Linux**:
    ```
    source env/bin/activate
    ```
    **Windows**:
    ```
    env\Scripts\activate
    ```
2. Launch Python main:<br>
    ```
    python3 main.py
    ```
3. Deactivate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/deactivate
    ```
    **Windows**
    ```
    env\Scripts\deactivate
    ```

## Run the Project on PyCharm

After creating the virtual environment, you can run the project on PyCharm.

**Steps:**

1. Open the project on PyCharm
2. Open the Terminal
3. Go to Settings -> Project -> Project Interpreter
4. Click on the Gear Icon and select Add Local
5. Select the virtual environment folder
6. Open main.py and run it

## General Development Workflow

1. Create a new local branch from the master branch
   ```
    git checkout -b <branch_name>
    ```
2. Make changes to the code on a separate python file
3. Once the feature is complete, start update the modified files on the local branch
4. Add the changes to the staging area
    ```
    git add <file_name>
    or to add all the files:
    git add .
    ```
5. Commit the changes
    ```
    git commit -m "<commit_message>"
    ```
6. Push the changes to the remote repository
    ```
    git push origin <branch_name>
    ```
7. Create a pull request on GitHub
8. Once the pull request is approved, merge the branch to the master branch
9. Delete the local branch
    ```
    git branch -d <branch_name>
    ```
10. Delete the remote branch
    ```
      git push origin --delete <branch_name>
    ```

# Current Workings
- [ ] Pass from Points Dataframe to &rarr; Stroke Dataframe
- [ ] Feature 1: Stroke Length


## License
This project is licensed under the @Università degli Studi di Cassino e del Lazio Meridionale

## Notes
On Windows:
Make sure to have Screen Scale at 100% instead of 120% or other values,
otherwise the task rendering on screen won't fit.