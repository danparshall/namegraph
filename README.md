namegraph
==============================

Building a directed graph of family relationships from name data

The dataset consists of public records from a country in Latin America, scraped off a government website.  Technically this isn't PII in the country of origin, but this repo will still treat it as such.  In Latin America, one's legal/formal last name (or apellido) consists of two parts: the father's last name (which is passed on to children) and the mother's last name (which is not); unlike in the USA, women don't change their legal name upon marriage.  This pattern gives us a chance to link together the records and establish a graph of family relationships.  In addition to the citizen's full legal name, we also typically have entries for the parents (although these are often in "social" format, rather than the "legal" format).  The goal of this repo is to parse the records, for each citizen identifying the apellidos, and linking each record to the parents.

For each citizen, we have the following information:
```
cedula	-	Basically a citizen ID number.  Hashed for privacy.
nombre	-	Citizen's name, almost always in the standard legal format
dt_birth	-
dt_death	- if applicable
marital_status	- Either SOLTERO, CASADO, DIVORCIADO, VIUDO (Single, Married, Divorced, Widowed)
dt_marriage
nombre_spouse
ced_spouse
nombre_padre
ced_padre
nombre_madre
ced_madre	
is_nat
is_nat_padre
is_nat_madre

```

The standard legal format for names is: `patronym matronym firstname middlename`.  This contrasts with the "social format" which has the prenames first.  The citizen's name is almost always in the legal format, but the other names vary widely.  Sometimes the record will be in standard legal form, but it's also common for the name to be `firstname patronym` (i.e., in social format, and only one surname).


# Example
Let's consider how the Simpsons family would look in this data.  The family tree is:
(./references/simpsons_family.png)

Homer's parents are Abraham Simpson & Mona Olsen, so Homer's apellidos would be "Simpson Olsen", and his full name would be `Homer Jay Simpson Olsen`

Marge's parents are Clancy Bouvier & Jacqueline Gurney.  In Spanish-style naming, Marge would retain her maiden name (Bouvier), and pass it on to her children.  Socially, she might be known as "Marge de Simpson", but on legal documents she would be `Marjorie Jacqueline Bouvier Gurney`.

|nombre	|	dt_birth	|	nombre_padre	|	nombre_madre	|	marital_status	|	dt_marriage	|	nombre_spouse
Simpson Olsen Homer Jay	|	1956/05/12	| Abe Simpson	|	Mona Olsen	|	CASADO	|	1981/09/29	|	Marge Bouvier
Bouvier Gurney Marjorie Jacqueline	|	1956/10/01	|	Clancy Bouvier	|	Jacqueline Gurney	|	CASADO	|	1981/09/29	|	Simpson Homer
Bart Jojo Simpson Bouvier	|	1981/04/01
Lisa Marie Simpson Bouvier	| 1983/05/09
Margaret Evelyn Simpson Bouvier


Chart taken from

https://web.archive.org/web/20210627221809/https://www.pngkey.com/maxpic/u2q8e6u2q8r5u2r5/
https://simpsons.fandom.com/wiki/Simpson_family?file=Simpsons_possible_family_tree.jpg

https://web.archive.org/web/20210628002600/https://simpsons.fandom.com/wiki/Simpson_family?file=Simpsons_possible_family_tree.jpg




===============================


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
