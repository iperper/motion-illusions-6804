# motion-illusions-6804

### Dependencies

Install Jupyter Notebook
```bash
python3 -m pip install jupyter
```

### Setup

Create a virtual environment to run the notebook

```bash
python -m venv virtual-motion
source virtual-motion/bin/activate

pip install --upgrade pip

pip install ipykernel numpy pillow opencv-python matplotlib

# Sets up the virtual environment to be accessible by the jupyter notebook
# See last step for using it.
python -m ipykernel install --user --name=virtual_motion
```

Run the jupyter notebook

```bash
jupyter notebook
```

Make sure to select the `virtual_motion` kernel when creating an new notebook. You can also switch the kernel in existing notebooks.
