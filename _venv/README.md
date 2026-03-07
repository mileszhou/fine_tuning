# Folder Structure
```
node/
├── _outputs/          # logs, plots, intermediate results
├── _results/          # processed experiments
├── src/               # source code / recipes
│   ├── venv/          # env setup scripts
│   └── lora_training/ # main ML logic
├── res/               # configuration files and supporting assets
│   ├── config/
│   ├── data/
│   └── third_party/
├── lib/               # shared libraries
│   └── helpers/
├── bin/               # internal helper scripts
│   └── run_scripts/
├── python/            # consumable scripts
│   └── train.py
├── pytorch/           # consumable trained models
│   └── model.pt
├── notebook/          # consumable notebooks
│   └── experiment.ipynb
└── vsbasic/           # optional consumable artifacts for other languages
    └── example.vb
```
This structure organizes the project into clear sections for outputs, source code, resources, libraries, scripts, and consumable artifacts. It allows for easy navigation and maintenance while keeping related files together.
# README
This project is structured to facilitate efficient development and organization of machine learning experiments. The `src/venv` directory contains scripts for setting up the virtual environment, ensuring that all dependencies are managed consistently across different setups. The `src/lora_training` directory houses the main logic for training models using LoRA (Low-Rank Adaptation), which is a technique for fine-tuning large language models efficiently. The `res/` directory includes configuration files, datasets, and third-party resources that are essential for the experiments. The `lib/helpers` directory contains shared libraries that can be used across different parts of the project, while the `bin/run_scripts` directory includes internal helper scripts for automating various tasks. The `python/` directory is reserved for consumable scripts that can be executed directly, such as training scripts or evaluation scripts. The `pytorch/` directory is where trained models are stored, and the `notebook/` directory contains Jupyter notebooks for exploratory data analysis and experiment documentation. Finally, the `vsbasic/` directory is an optional section for any consumable artifacts that may be relevant for other programming languages, such as Visual Basic examples. This structure is designed to promote modularity, reusability, and clarity, making it easier for developers to navigate and contribute to the project.
# Setup Instructions
1. **Clone the Repository**: Start by cloning the repository to your local machine using the following command:
   ```
   git clone    <repository_url>
   ```  
2. **Navigate to the Project Directory**: Change into the project directory:
   ```
    cd node
    ```
3. **Set Up the Virtual Environment**: Use the provided scripts in the `src/venv` directory to set up your virtual environment. This will ensure that all necessary dependencies are installed and managed properly. For example, you can run:
4.   ```
    bash src/venv/setup.sh
    ```
5. **Activate the Virtual Environment**: After setting up, activate the virtual environment to start working on the project:
6.  ```
    source venv/bin/activate
    ```
7. **Install Additional Dependencies**: If there are any additional dependencies required for specific experiments, you can install them using pip. For example:    
   ```
    pip install -r res/config/requirements.txt
    ```
8. **Run Training Scripts**: You can now run the training scripts located in the `python/` directory. For example, to start training a model, you can execute:
   ```
    python python/train.py --config res/config/train_config.yaml
    ```
9. **Monitor Outputs**: The outputs of your experiments, including logs and intermediate results, will be stored in the `_outputs/` directory. You can monitor this directory to track the progress of your experiments.
10. **Save Results**: Once your experiments are complete, processed results can be saved in the `_results/` directory for further analysis and reporting.
11. **Explore Notebooks**: You can also explore the Jupyter notebooks in the `notebook/` directory for data analysis and experiment documentation. To launch a notebook, use the following command:
    ```    jupyter notebook notebook/experiment.ipynb
    ```
12. **Contributing**: If you wish to contribute to the project, please follow the standard Git workflow. Create a new branch for your feature or bug fix, make your changes, and submit a pull request for review. Ensure that your code adheres to the project's coding standards and includes appropriate documentation and tests.

