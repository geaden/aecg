# Adaptive Variants of Frank-Wolfe Method with Relative Inexact Gradient Information

This repository contains the code for the paper:

> Adaptive Variants of Frank-Wolfe Method with Relative Inexact Gradient Information,
>
> Gennady Denisov<sup>1</sup>, Fedor Sergeevich Stonyakin<sup>1,2</sup> and Mohammad Alkousa<sup>3</sup>
>
> <sup>1</sup> Moscow Institute of Physics and Technology, Russia
>
> <sup>2</sup> V.,I.,Vernadsky Crimean Federal University, Russia
>
> <sup>3</sup> Innopolis University, Russia

[Paper](https://doi.org/10.1007/978-3-031-97077-1_1)

```bibtex
@InProceedings{10.1007/978-3-031-97077-1_1,
    author="Denisov, Gennady
    and Stonyakin, Fedor
    and Alkousa, Mohammad",
    editor="Kochetov, Yury
    and Khachay, Michael
    and Eremeev, Anton
    and Pardalos, Panos",
    title="Adaptive Variants of Frank-Wolfe Method with Relative Inexact Gradient Information",
    booktitle="Mathematical Optimization Theory and Operations Research",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="3--16",
    abstract="The article introduces the adaptive versions of the Erroneous Conditional Gradient (ECG) algorithm with an Erroneous Oracle (EO) and a Linear Minimization Oracle (LMO) on a box-constrained feasible set. Two step-size strategies are studied: the first one displaying a dependency on the iteration, while the second one depends on the L-smoothness constant. This paper highlights the results of the implementation of these algorithms tested through computational experiments. PageRank is chosen for the algorithms to be applied to the optimization problem since the complexity of the former remains relevant even nowadays. The quality of the solution aligns with the theoretical expectations. Further research and practical implications of these algorithms are discussed in the conclusion.",
    isbn="978-3-031-97077-1"
}
```


## Development

To run numerical experiments:

```bash
python3 -m venv .venv
pip install -r requirements.txt
```

and open [`aecg.ipynb`](aecg.ipynb) in Jupyter Notebook or in VS Code.

Testing:

```bash
pytest .
```
