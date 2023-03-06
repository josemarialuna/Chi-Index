<a name="readme-top"></a>
# External Clustering Validation Chi Index
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

[![PyPI](https://img.shields.io/pypi/v/ADLStream.svg)](https://pypi.org/project/chi-index/)
[![Downloads](https://static.pepy.tech/badge/chi-index)](https://pepy.tech/project/chi-index)
[![Python 3.6](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About Chi Index
Chi Index is an external clustering validity index that measure the distance between the instances of a clustering result and the labels. Although clustering is an unsupervised learning machine learning technique, Chi index favors that the clusters formed have the least number of different labels.

For example, in the following image we can see 3 different clustering solutions, in which each of the circles represents an instance of the dataset, and the color, the class to which it belongs. In A, we can see that there is a cluster that has 5 red instances, and two green instances, while in the other cluster we have 2 red instances, 8 green instances, and 6 blue instances. In solution B, with k=3, we find that the cluster at the top of the figure has mostly red instances, the one on the left is mostly blue, and the one at the bottom has mostly green instances.

<p align="center">
  <img alt="Clustering Solutions" src="https://github.com/josemarialuna/Chi-Index/blob/6cca612637ee3f9eea38bb4738a58007a2de86e5/images/chi-solutions.jpg" width="60%">
</p>

Chi index measures the distribution of instances from the clusters formed and the number of instances of each label in them, and calculates a metric based on the chi-square statistic. In the following table we can see the chi index results for each of the clustering solutions. 

| k  | Chi Index(k) |
|:-------------: |:-------------:|
| 2        |  	0.890        |
| 3         |  **0.925**        |
| 4        |  0.760        |

As we can see, the clustering solution with the highest chi index value is k=3, which indicates that to separate instances of the same label into clusters, the optimal number of clusters is 3.

The higher the chi index value, the greater the dependency between clusters and labels, i.e. the clustering solution with the highest chi index will indicate that the instances belonging to the same class are grouped as well as possible in the clusters.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started
Using Chi Index is very simple, and here is how to do it in a few steps. You just need to have installed the Chi Index library available through the pip, and after that, you will need to import it into your Python application.

### Installing Chi Index

The Chi index version of this repository is implemented in Python. You can use any version of Python from 3.7 onwards, although it is recommended to use 3.10. To install the library you only need to execute the following command:
```bash
pip install chi-index
```

### Example

To run this example you must have installed the chi index library by executing the command in the previous section. 
After that, you must download the file iris.data from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/iris), and place it in a folder called "data". To make it easier for you, I leave here the link: [iris.data]([http://www.limni.net](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data))

Then you can copy and paste the following code:

```python 
import pandas as pd
from chi_index.model import ChiIndex


def main():
    df = pd.read_csv('./test/data/iris.data', delimiter=",", header=None)
    print(df.columns)
    print(df.head())
    df.rename(columns={4: 'Class'}, inplace=True)

    chi = ChiIndex(df, results_path='result')
    print(chi.list_chi)
    print(chi.optimum_chi)
    print(chi.optimum_k)
    chi.save_centroids()


if __name__ == "__main__":
    main()
```

If you have any problem, or you don't manage to execute the code, please contact me through [DISCUSSION](https://github.com/josemarialuna/Chi-Index/discussions) so I can help you.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Read [CONTRIBUTING.md](CONTRIBUTING.md). We appreciate all kinds of help.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact 

* **José María Luna-Romera** - [LinkedIn](http://www.linkedin.com/in/josemluna)
* **José C. Riquelme**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Cite this
Please, cite as: Luna-Romera JM, Martínez-Ballesteros M, García-Gutiérrez J, Riquelme JC. External clustering validity index based on chi-squared statistical test. Information Sciences (2019) 487: 1-17. https://doi.org/10.1016/j.ins.2019.02.046. (http://www.sciencedirect.com/science/article/pii/S0020025519301550)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/josemarialuna/Chi-Index.svg?style=for-the-badge
[contributors-url]: https://github.com/josemarialuna/Chi-Index/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/josemarialuna/Chi-Index.svg?style=for-the-badge
[forks-url]: https://github.com/josemarialuna/Chi-Index/network/members
[stars-shield]: https://img.shields.io/github/stars/josemarialuna/Chi-Index.svg?style=for-the-badge
[stars-url]: https://github.com/josemarialuna/Chi-Index/stargazers
[issues-shield]: https://img.shields.io/github/issues/josemarialuna/Chi-Index.svg?style=for-the-badge
[issues-url]: https://github.com/josemarialuna/Chi-Index/issues
[license-shield]: https://img.shields.io/github/license/josemarialuna/Chi-Index.svg?style=for-the-badge
[license-url]: https://github.com/josemarialuna/Chi-Index/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/josemarialuna
[product-screenshot]: images/screenshot.png
 
