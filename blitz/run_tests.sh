eval "$(conda shell.bash hook)"
conda activate blitz;
pip install ../.;
python -m unittest discover -p '*_test.py' -s '.'
