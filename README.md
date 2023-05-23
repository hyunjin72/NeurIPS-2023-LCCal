## Running the Code
- Create a virtual environment
```
virtualenv pevenv --python=python3.7
. pevenv/bin/activate
```
- Install required packages
```
pip install -r requirements.txt
```
- To reproduce the calibration performance of LCCal (GAT) on Cora, run the following script:
```
python main.py --dataset Cora --model GAT --calibration 'LCCT' --wdecay 5e-4 --b_over 0.1 --b_under 0 --default_l 0.35
```