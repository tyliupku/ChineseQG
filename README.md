# ChineseQG
This project includes two ways to generate simple questions based on RDF triples in the knowledge base, template extraction method and template based sequence-to-sequence(seq2seq) model.

## Model Overview

### Template Extraction
<p align="center"><img width="80%" src="doc/template-extraction.png"/></p>
As shown above, we extract templates for a specific predicate ("相关人物"/related people in this case) from training set. Then we randomly select one of those extracted templates to generate new questions of the given triples from testing set.   

### Template-based Seq2seq
<p align="center"><img width="70%" src="doc/tseq2seq.png"/></p>
For a given triple, the input for triple encoder is the concatenation of (SUBJECT, SEP, PREDICT). SEP here is '|||'.

For template decoder, instead of using the entire question for input, we replace the SUBJECT(topic words) in the question with '(SUB)' token. 

## Installation
For training tseq2seq, we strongly recommend using GPU for accelerating the speed.

### Tensorflow
The code for tseq2seq is based on Tensorflow. You can find installation instructions [here] ()
