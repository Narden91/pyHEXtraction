# pyHEXtraction

## Prerequisites after first pull

1. Create folder for data:
    ```
    create a folder called "data"
    ```
2. Create Python Virtual Environment:
    ```
    python -m venv env
    ```
3. Activate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/activate
    ```
    **Windows**
    ```
    env\Scripts\activate
    ```
4. Load Required libraries
    ```
    pip install -r requirements.txt
    ```
5. Give Script permissions in Linux:<br>
    ```
    chmod +x main.py
    ```
6. Deactivate Virtual Environment:<br>
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