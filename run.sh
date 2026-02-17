#!/bin/bash
python -m magnet.evaluation jhu_ta1/cards/jhu_instance_predict_auc.yaml

# To run on a different dataset:
# python -m magnet.evaluation jhu_ta1/cards/jhu_instance_predict_auc.yaml --set dataset=legalbench:subset=abercrombie
