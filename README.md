# CiteRAG

CiteRag is an early experimental Tool that utilizes OpenAIs GPT and Embedding models to help conduct literature reviews.

## Installation

Simply download the appropriate variant of the script for which you want to use either an OpenAI API key or an Azure OpenAI API key.

## Usage

(1) Search for suitable literature for your research in an appropriate database.

(2) Download the pdf files of the publications.

(3) Customize the script:
```python
### Fill in your OpenAI API Key (https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key) ###
oakey="<YOUR API KEY GOES HERE>"
### SET YOUR PATH TO YOUR FOLDER CONTAINING YOUR PDFs HERE ###
directory="<YOUR PATH GOES HERE>"
```

(4) Set up your research questions:
```python
query1="Question 1 (e.g. "Is this paper related to a specific topic?)""
query2="Question 2"
query3="Question 3"
query4="Question 4"
query5="Question 5"
query6="Question 6"
```
(5) Run the script

## Please Note

It is possible to ask more or less than the predefined six questions. However, further adjustments must then be made in the script. 

CiteRAG is still under development and is not a finished end product. 

Due to the large number of API requests, a timeout error may be thrown if a large number of pdf files are requested. In this case, make sure that the intermediate results are saved and restart the program after adjusting the for loop.

## License

[CC BY-NC-ND 4.0](https://github.com/dhefft/CiteRAG/blob/main/LICENSE.md)

## Citation

Please cite this work as:
tbd
