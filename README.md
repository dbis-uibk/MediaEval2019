# Emotion and Theme Recognition in Music Using Jamendo
This repository contains the implementation of our, team TaiInn (Innsbruck),
solution to the
[Emotion and Theme Recognition in Music Using Jamendo](https://multimediaeval.github.io/2019-Emotion-and-Theme-Recognition-in-Music-Task/)
task as part of [MediaEval 2019](http://www.multimediaeval.org/mediaeval2019/).

## Repository Structure
The `results` folder contains the predictions, decisions and results of our
five submitted runs.  The according source code can be found in the `src`
folder and is under BSD-2-Clause license.

## Examples
Setup the pipenv defined in the `src` folder using `pipenv install` and then run:
```
pipenv run python -m dbispipeline configs/crnn_16.py --path=$HOME/mediaeval2019_data
```
