eval "$(conda shell.bash hook)"
conda activate torchenv;
python -m unittest discover -p '*_test.py'
