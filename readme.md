# Session-based-recommenders

This repository was forked from the [rn5l/session-rec](https://github.com/rn5l/session-rec) repo. Our goal was to provide a script to run experiments and provide some benchmarks for different recommendation algorithms, mainly we were interested in benchmarking the CSRM, SR-GNN, and the NARM recommenders.We test our result on retailrocket dataset and diginetica dataset.
<div>
<table class="table table-hover table-bordered">
    <tr>
        <th width="20%" scope="col"> Algorithm</th>
        <th width="12%" class="conf" scope="col">File</th>
        <th width="68%" class="conf" scope="col">Description</th>
    </tr>
    <tr>
        <td scope="row">CSRM</td>
        <td>csrm.py</td>
        <td>Wang et al., A collaborative session-based recommendation approach with parallel memory modules, SIGIR 2019.<br>
        </td>
    </tr>
    <tr>
        <td scope="row">NARM</td>
        <td>narm.py</td>
        <td>Li et al., Neural Attentive Session-based Recommendation, CIKM 2017.
        </td>
    </tr>
    <tr>
        <td scope="row">SR-GNN</td>
        <td>gnn.py</td>
        <td>Wu et al., Session-based recommendation with graph neural networks, AAAI 2019.
        </td>
    </tr>
</table>
</div>

<!-- Deadline: 12/15/2022 -->
## Task

Train different session (contextual, sequential) based product recommendation
recommenders for E-commerce use case and compare the performance of the recommenders.


## Result

<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Recommender</th>
            <th width="16%" class="conf" scope="col">Dataset</th>
            <th width="16%" class="conf" scope="col">HitRate@20</th>
            <th width="16%" class="conf" scope="col">MRR@20</th>
            <th width="16%" class="conf" scope="col">MAP@20</th>
            <!-- <th width="16%" class="conf" scope="col">HitRate</th> -->
        </tr>
        <tr>
            <td>SR-GNN</td>
            <td>RetailRocket</td>
            <td>0.481872818922062
            <td>0.261930399859539
            <td>0.250933278906411
            </td>
        </tr>
        <tr>
            <td>SR-GNN</td>
            <td>Diginetica</td>
            <td>0.337710701685613
            <td>0.118414756337152
            <td>0.107449959069729
            </td>
        </tr>
        <tr>
            <td>CSRM</td>
            <td>RetailRocket</td>
            <td>0.5282086079875921
            <td>0.28017071917987535
            <td>0.2677688247394867
            </td>
        </tr>
        <tr>
            <td>CSRM</td>
            <td>Diginetica</td>
            <td>0.392272638181105
            <td>0.130038309859531
            <td>0.116926593443457
            </td>
        </tr>
        <tr>
            <td>NARM</td>
            <td>RetailRocket</td>
            <td>0.559034509499806
            <td>0.307978499374293
            <td>0.295425698868014
            </td>
        </tr>
        <tr>
            <td>NARM</td>
            <td>Diginetica</td>
            <td>0.437941003528028
            <td>0.152351861021737
            <td>0.138072403896433
            </td>
        </tr>
    </table>
</div>

## Getting Started

### Requirement

Linux with Python ≥ 3.6

CUDA 11.6

An NVIDIA GPU

### Installation

1. Download and Install Anaconda (https://www.anaconda.com/distribution/)
2. Go to our session-based-recommenders repository
3. Run installation:
    ```
    conda env create -f env_gpu.yaml
    ```
4. Activate the conda environment: 
    ```
    conda activate srec37
    ```
5. Datasets can be downloaded from: https://drive.google.com/drive/folders/1ritDnO_Zc6DFEU6UND9C8VCisT0ETVp5?usp=sharing

Download retailrocket dataset and diginetica dataset.

6. Unzip file and move it to data folder.

diginetica:

train-item-views.csv should be placed in `data/diginetica/raw`

retailrocket:

events.csv should be placed in `data/retailrocket/raw`


7. Preprocess data by running a configuration with the following command:
retailrocket:
    ```
    python run_preprocessing.py conf/preprocess/session_based/window/retailrocket.yml 
    ```
diginetica:
    ```
    python run_preprocessing.py conf/preprocess/session_based/window/diginetica.yml
    ```
8. Run training and evaluation
retailrocket:
    ```
    THEANO_FLAGS="device=cuda0,floatX=float32" CUDA_DEVICE_ORDER=PCI_BUS_ID python run_config.py conf/exp_retailrocket_models.yml conf/out
    ```
diginetica:
    ```
    THEANO_FLAGS="device=cuda0,floatX=float32" CUDA_DEVICE_ORDER=PCI_BUS_ID python run_config.py conf/exp_diginetica_models.yml conf/out
    ```
<!-- Example of configuration
```
- class: emde.model.EMDE
  params: {dataset: retailrocket, alpha: 0.9, W: 0.01, bs: 256, lr: 0.004, gamma: 0.5, n_sketches: 10,
          sketch_dim: 128, hidden_size: 2986, num_epochs: 5,
          slice_absolute_codes_filenames: ['data/retailrocket/codes/slices/SessionId_iter2_dim1024',
                                          'data/retailrocket/codes/slices/SessionId_iter4_dim1024',
                                          'data/retailrocket/codes/slices/UserId_iter3_dim1024'],
          master_data_absolute_codes_filenames: ['data/retailrocket/codes/mm/property_6',
                                                'data/retailrocket/codes/mm/property_776',
                                                'data/retailrocket/codes/mm/property_839',
                                                'data/retailrocket/codes/mm/random'],
          evaluate_from_dataLoader: True
  }
  key: emde
```

`dataset` - name of dataset

`alpha` - defines time decay in history user's sketch `sketch(t2) =alpha*W^(time_diff)*sketch(t1)`

`W` - defines time decay in history user's sketch `sketch(t2) =alpha*W^(time_diff)*sketch(t1)`

`bs` - training batch size

`lr` - learning rate

`gamma` learning rate decay after each epoch

`n_sketches` sketch depth

`sketch_dim` sketch width

`hidden_size` hidden size of feed forward neural network

`num_epochs` number of epochs

`slice_absolute_codes_filenames` list of json filename with product codes, seperate filenames per slice with extension `.{slice_number}`


`master_data_absolute_codes_filenames` list of json filename with product codes, common from all slices

`evaluate_from_dataLoader` If True evalues using pytorch dataLoader else using `predict_next` method -->

