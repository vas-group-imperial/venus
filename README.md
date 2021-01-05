Requirements 
------------

* Python 3.7 or higher 
* Gurobi 8.1 or higher. 
* Keras 2.2 or higher with TensorFlow backend.

Installation
------------

## Install Gurobi

```sh
wget https://packages.gurobi.com/9.0/gurobi9.0.1_linux64.tar.gz
tar xvfz gurobi9.0.1_linux64.tar.gz
cd gurobi901/linux64
python3 setup.py install
```

#### Add the following to the .bashrc file:
```sh
export GUROBI_HOME="Current_directory/gurobi901/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

#### Retrieve a Gurobi License

To run Gurobi one needs to obtain license from [here](https://www.gurobi.com/documentation/9.0/quickstart_linux/retrieving_and_setting_up_.html#section:RetrieveLicense).

## Install python dependencies

```sh
pip3 install -r requirements.txt
```

Usage
-------------

## ACAS 

```sh
python3 . --net <path> --property acas --acas_prop <property number>
```

## MNIST/CIFAR-10

```sh
python3 . --net <path> --property lrob --lrob_input <path> --lrob_radius <real>
```

## Arguments

| Argument | Values | Description |
| -------- | ------ | ----------- |
| net | Path | Path to the neural network in Keras format |
| property | {acas, lrob} | Verification property, one of acas or lrob (local robustness) |
| acas_prop | Integer from 0 to 10 | Acas property to check. Default value is 1 |
| lrob_input | Path | Path to a pickle object of a pair whose first component is the network's input and its second component is the label of the input |
| lrob_radius | Real | Perturbation radius for L_inifinity norm. Default value is 0.1 |
| st_ratio | Real | Cutoff stability ratio of the splitting procedure. Default value is 0.5 |
| depth_power | Reak | Controls the splitting depth. Higher values favour splitting. Default value is 1 |
| splitters | Integer | Number of splitting processes = 2^splitters. Default value is 0 |
| workers | Integer | Number of worker processes. Default value is 1 |
| offline_dep | Boolean | Controls whether to include offline dependency cuts (before starting the solver) or not. Default value is True |
| online_dep | Boolean | Controls wether to include online dependency cuts (through solver callbacks) or not. Default value is True |
| ideal_cuts | Boolean | Controls whether to include online ideal cuts (through solver callbacks) or not. Default value is True |
| opt_bounds | Boolean | Controls whether to optimise bounds using linear relaxation or not. Default value is False |
| timeout | Integer | Timeout in seconds. Default value is 3600. |
| logfile | Path | Path to logging file. |
| print | Boolean | Controls extra information or not. Default value is False. |

Examples
-------------

An AcasXU property, one of those defined in Katz et al, 2017, can be verified with the following command:
```
python3 . --property acas --acas_prop 0 --net ./resources/acas/models/acas_1_1.h5 --st_ratio 0.7 --depth_power 20 --splitters 1 --workers 4 --offline_dep False --online_dep False --ideal_cuts False
```

Here we verify Property 1 over the acas_1_1.h5 network using optimal parameters for this case. Namely, we favour input splitting until a high percentage of ReLU nodes is stable (around 70%). Venus will print the following:

* Information about the property and the network to be verified.

* 'Satisfied' when the property holds and 'NOT Satis' when it manages to find a counter-example showing that the property does not hold. In the latter case it also prints the counter-example.

* Verification time in seconds.

* Number of subproblems solved by  Gurobi. When the number is 0, it means that it was possible to prove the property by preprocessing, namely by checking the output bounds against the condition on the output layer. 

The mapping between properties and the numbers are as follows:

* 0 corresponds to Property 1, 1 to Property 2, ..., 4 to Property 5.
* Property 6 has a union of two input regions, so 5 corresponds to Property 6 on the first region, and 6 to Property 6 on the second region.
* 7 corresponds to Property 7, 8 to Property 8, 9 to Property 9 and 10 to Property 10.
 

A local robustness property can be verified as follows:
```
python3 . --property lrob --lrob_input ./resources/mnist/evaluation_images/im1.pkl --lrob_radius 0.05 --net ./resources/mnist/mnist-net.h5 
```

Here we are checking local robustness of an MNIST fully connected network for a particular input image stored together with its correct classification label in pickle format and for a perturbation radius of 0.05. Checking local robustenss for a CIFAR10 image and network is done analogously. 

Similarly to AcasXU properties, Venus will output whether the property has been verified along with the timing.


Publications
-------------

*  [Efficient verification of ReLU-based neural networks via dependency analysis](https://vas.doc.ic.ac.uk/papers/20/aaai20-dep-analysis.pdf).
    Elena Botoeva, Panagiotis Kouvaros, Jan Kronqvist, Alessio Lomuscio, and Ruth Misener. 
    AAAI 2020.


Resources
---------------

In the `resources` folder we provide the networks and the MNIST and CIFAR10 images that we used for the experiments in the paper above. The feed-forward fully-connected ReLU-activated MNIST and CIFAR10 networks have been trained with the Tensorflow framework using the standard training procedures. The AcasXU networks have been obtained by converting the `.nnet` networks from the [Reluplex repository](https://github.com/guykatzz/ReluplexCav2017) into the Keras format. 

The files are structured into three sub-folders, one for each of the benchmarks we have used for the experiments: `acas`, `mnist` and `cifar`. 

* the `acas` folder contains the file `acasprop.py` with a description of the 10 AcasXU properties as defined in Katz et al, 2017, and the folder `models` with the 45 AcasXU networks in the Keras format. The `acasprop.py` file is being used by `__main__.py` to map between a number from 0 to 10 the an actual AcasXU property. 

* the `mnist` folder contains the network file `mnist-net.h5` in the Keras format and the folder `evaluation_images` with 100 randomly selected MNIST images that are correctly classified by the network, stored together with their classification label in pickle format `im{i}.pkl`. The network has two hidden ReLU-activated layers of 512 nodes each and the final layer is activated by the linear function.

* likewise, the `cifar` folder contains the network file `cifar-net.h5` in the Keras format and the folder `evaluation_images` with 100 randomly selected CIFAR10 images. The network has three hidden ReLU-activated layers of 1024, 512 and 512 nodes, respectively, and the final layer is activated by the linear function.

Additionally, in the `evaluation` folder we provide three bash scripts for verifying all AcasXU properties and for verifying local robustness for each of the 100 images, both in the MNIST and the CIFAR10 use-case. Running, for instance, `test_acas_properties.sh` will call the verifier for each of the AcasXU properties with a timeout of 1 hour, and store the verification results in the `acas.log` file.


Experimental Results
--------------
We ran our experiments on an Intel Core i7-7700K (4 cores) with a main memory of 16GB, operating Ubuntu 18.04. We compared Venus against complete verifiers such as [Marabou](https://github.com/NeuralNetworkVerification/Marabou), [Neurify](https://github.com/tcwangshiqi-columbia/Neurify/) (for the AcasXU experiments we used [ReluVal](https://github.com/tcwangshiqi-columbia/ReluVal/)), and finally, against [NSVerify](https://vas.doc.ic.ac.uk/software/neural/). 

We compare the performance of all 4 systems in terms of the number of problems that could be solved within the timeout of 1 hour, the total verification time including timeouts and the total verification time for the problems that the three best performing tools were able to solve.

We obtained the following results for local robustness properties for the 100 randomly selected MNIST images with perturbation radius of 0.05.


| Tool | Solved | Total time | Time solved |
| ---- | ------ | ---------- | ----------- |
| Venus | 100 | 5,953.46 | 573.38 | 
| Marabou | 0 | 86,400.00 | -- |
| Neurify | 65 | 126,007.24 | 7.00 |
| NSVerify | 95 | 26,906.81 | 2,515.15 |


We run Venus with 2 worker processes and dependency analyser turned on (for more detailed information, refer to the paper). For Marabou we used the parameters reported in Katz et al, 2019: T=5, k=4, and m=1.5, and we employed 4 worker processes. For Neurify we set the MAX_THREAD parameter to 1 (which should result in 2 parallel processes). We run NSVerify with default parameters.

Nest, we verified local robustness properties for the 100 randomly selected CIFAR10 images with perturbation radius of 0.01 and obtained the following results.


| Tool | Solved | Total time | Time solved |
| ---- | ------ | ---------- | ----------- |
| Venus | 100 | 560.04 | 7.36 | 
| Marabou | 0 | 86,400.00 | -- |
| Neurify | 76 | 778.46 | 10.24 |
| NSVerify | 100 | 3,460.41 | 45.53 |


For all systems we used exactly the same parameters as for MNIST experiments.

Finally, we verified the 172 AcasXU properties as defined in Katz et al, 2017, and obtained the following results.


| Tool | Solved | Total time | Time solved |
| ---- | ------ | ---------- | ----------- |
| Venus | 170 | 19,642.57 | 5,527.76 | 
| Marabou | 156 |  140,916.96 | 75,747.78 |
| Neurify | 167 | 23,628.75 | 2,555.38 |
| NSVerify | 6 | 86,400.00 | -- |


We run Venus with 4 worker processes, dependency analyser turned off, stability ratio set to 0.7 and depth power parameter to 20 (for more detailed information, refer to the paper). For Marabou we used the same parameters as before. For Neurify we set the MAX_THREAD parameter to 2 (which should result in 4 parallel processes). We run NSVerify with default parameters.

As we can see, in all three cases Venus was able to prove or disprove the highest number of properties and was fastest in terms of the total verification time (including timeouts).

We conclude with a graphical representation of the total number of verification queries that each tool could verify as a function of time.

![High Level](https://vas.doc.ic.ac.uk/papers/20/nsolved.png)

More experimental results can be found in our papers.

Contributors
--------------

* Panagiotis Kouvaros (lead contact) - p.kouvaros@imperial.ac.uk

* Elena Botoeva - e.botoeva@imperial.ac.uk

* Alessio Lomuscio - a.lomuscio@imperial.ac.uk

License and Copyright
---------------------

* Licensed under the [BSD-2-Clause](https://opensource.org/licenses/BSD-2-Clause)
