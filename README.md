### Paper ###

This is the companion library for the papers:

(short paper) *Naive Bayes Classification for Subset Selection in a Multi-label Setting*, Mossina and Rachelson, ICPRAI (2018) <br>
[http://oatao.univ-toulouse.fr/19557/](http://oatao.univ-toulouse.fr/19557/) <br>
(long version) * Naive Bayes Classification for Subset Selection*, Mossina and Rachelson, arXiv preprint (2017) <br>
[https://arxiv.org/abs/1707.06142](https://arxiv.org/abs/1707.06142)


### Test info ###
The `test/main.cpp` file provides a script to run NaiBX.

As commented in the source, one must explicitly provide some input parameter, like the number of columns (e.g. `dim_x=103`) and the number of target labels available (e.g. `labels=14`). Please, for more information read the source code.


### Example of use ###
>`$ make naibx`, will get you started

Data are assumed to be cleaned and ready for the analysis, that is, columns are either features (explicative variables) or labels.

The authors of [mulan](http://mulan.sourceforge.net/datasets-mlc.html) and [MEKA](https://sourceforge.net/projects/meka/files/Datasets/) very conveniently provide the community with a collection of multi-labeled datasets.

As an example, download the dataset [`yeast.arff`](https://sourceforge.net/projects/meka/files/Datasets/Music.arff/download) from  and run:
> `$ ./naibx data=path/to/yeast.arff dim_x=103 labels=14 all`

Input parameters
- `data=`: path to your copy of `yeast.arff`
- `dim_x=`: number of features. This assumes
- `all`: computes all metrics available

### *Bag-of-words* Features ###
> `$ ./naibx -bow data=/path/to/data.arff dim_x=103 labels=14`

`./naibx` allows you to do k-fold cross-validation automatically, by providing the number of folds (e.g. `kfold=10`).
