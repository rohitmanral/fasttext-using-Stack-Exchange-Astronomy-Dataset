# fasttext-using-Stack-Exchange-Astronomy-Dataset


<center> <h1>fastText Implementation </h1> </center>

<p align="center">
  <img width="260" height="150" src="https://fasttext.cc/img/ogimage.png">
</p>


This repository implemented fastText on Astronomy dataset, which is an open-source, lightweight, free library that allows users to learn text classifiers and text representations. It works on standard, generic hardware. Model size can later be reduced to fit on mobile devices.

* Environment : AWS Virtual Machine and server - Ubuntu 18.04 LTS , Google Colaboratory
* Code Management (Version Control, Code History): GitHub
* For Reporting: Latex + Overleaf

## Astronomy Stack Exchange Data
Astronomy Stack Exchange is a question and answer site for astronomers and astrophysicists. The dataset has been obtained from [Astronomy Stack Exchange](https://astronomy.stackexchange.com/questions?tab=newest&pagesize=50). I have used Python's Library called BeautifulSoup to scrape Questions and their respective Tags from the webpages. This dataset contains 9800 rows and some labels and is a multi label dataset. I have also explored hierarchical softmax and bigrams for this dataset.
