eval "$(conda shell.bash hook)"
conda activate blitz;
python -m unittest discover -p '*_test.py' -s '.'
