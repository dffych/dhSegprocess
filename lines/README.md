### Evaluation
``` python
python wk_evaluation.py \
--csv-filename ~/TM_trainings/LA_cBAD/tmp_eval/test_data.csv \
--model-dir /scratch/sofia/TM_tmp_models/LA_cBAD/export/1550677303/ \
--jar-tool-path ~/cDATA_LA_cBAD/TranskribusBaseLineEvaluationScheme_v0.1.3/TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar \
--output-export-dir /tmps/cbad_process/ \
--groundtruth-xml-dir ~/cDATA_LA_cBAD/GT_xml \
--resizing 0 \
--n_processes 8
--batch_prediction 1
```

Be careful if TextRegion has tag type="", the evaluation jar tool is not able to read the lines...