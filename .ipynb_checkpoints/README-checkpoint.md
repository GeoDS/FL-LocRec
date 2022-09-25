# FL-LocRec

A privacy‐preserving framework for location recommendation using federated learning
 
## Paper

If you find our code useful for your research, please cite our paper:

*Rao, J., Gao, S., Li, M., & Huang, Q. (2021). [A privacy‐preserving framework for location recommendation using decentralized collaborative machine learning](https://onlinelibrary.wiley.com/doi/abs/10.1111/tgis.12769). Transactions in GIS, 25(3), 1153-1175.*

```
@article{rao2021privacy,
  title={A privacy-preserving framework for location recommendation using decentralized collaborative machine learning},
  author={Rao, Jinmeng and Gao, Song and Li, Mingxiao and Huang, Qunying},
  journal={Transactions in GIS},
  volume={25},
  number={3},
  pages={1153--1175},
  year={2021},
  publisher={Wiley Online Library}
}
```

## Requirements

FL-LocRec uses the following packages with Python 3.6.13

```
pysyft==0.2.9
torch==1.4.0
numpy==1.18.5
av==8.0.3
pylibsrtp==0.7.1
scikit-learn==0.24.2
torchfm==0.7.0
```

## Usage

### Data

Raw data:

* nyc_checkin_500.csv - Foursquare check-in records in NYC.
* nyc_parking_lot.csv - locations of parking lots in NYC.
* nyc_bus_stop.csv - locations of bus stops in NYC.
* nyc_subway_station.csv - locations of subway stations in NYC.
* nyc_complaint.csv - locations of complaint records in NYC by NYPD.
* nyc_taxi_od.csv - origin-destination flows of taxi trips in NYC.

Encoded data:

* hex7/ - encoded check-in data (train/val/test/all) including all attributes at hex7 scale.
* hex8/ - encoded check-in data (train/val/test/all) including all attributes at hex8 scale.

### Training, Validation, and Testing

Train FL-LocRec using the encoded data and validate/test models.

```
python training.py [--train_path] [--val_path] [--test_path] [--save_path] [--batch_size] [--user_batch_size] [--num_epochs] [--embed_dim] [--mlp_dim] [--weight_decay] [--lr] [--dropout] [--precision_fractional] [--device]
```

### Dependency

We mainly referred to these awesome works:

PySyft by OpenMined [[Github]](https://github.com/OpenMined/PySyft)

PyTorch-FM by rixwew et al. [[Github]](https://github.com/rixwew/pytorch-fm)
