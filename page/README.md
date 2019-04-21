### Page experiment
Based on paper ["PageNet: Page Boundary Extraction in Historical Handwritten Documents."](https://dl.acm.org/citation.cfm?id=3151522)


#### Dataset 
The page annotations come from this [repository](https://github.com/ctensmeyer/pagenet/tree/master/annotations). We use READ-cBAD data with _annotator 1_ and _set1_.

`utils.page_dataset_generator` is used to generate the label images.


#### Usage

Prediction : use ``wk_process`` file in the folowing way :
```
python wk_process.py --filenames_to_predict data_to_predict/*.jpg \
                    --model-dir models/LA_page_model_batch_prediction \ 
                    --output-export-dir export_dir/ \
                    --n-processes 8
```

Evaluation : use ``wk_evaluation`` file in the following way :
```
python wk_evaluation.py --csv-filename test_data.csv \
                        --model-dir models/LA_page/LA_page_model_single/ \
                        --output-export-dir export_dir \
                        --groundtruth-json-dir GT_json_page_borders/ \
```
