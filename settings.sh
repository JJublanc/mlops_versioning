conda create --name ds_versioning python="3.7" jupyter
conda activate ds_versioning
pip install -r requirements.txt --user
ipython kernel install --name ds_versioning --user