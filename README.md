# <p align=center>`A Multi-Document Coverage Reward for RELAXed Multi-Document Summarization`</p>

## Code
`scripts/alternate_objectives.py`: Contains metrics used during training, and architecture for RELAX and gumbel sampling.

`scripts/xyz_dataset.py`: HuggingFace Datasets loader scripts (for fine-tuning).

`scripts/summarization.py`: Main file with model forward pass.

Links to the pre-trained Longformer models can be found in the `Longformer` repo.
You can download the WCEP dataset directly from [here](https://github.com/complementizer/wcep-mds-dataset).

### Run Code
```
# Run NLL pre-training
python3 scripts/summarization.py --save_dir='multinews' --model_path='longformer-encdec-large-16384' --grad_ckpt --max_output_len 256 --dataset='multinews' --num_dataset_examples='all' --epochs 5
```

```
# Run fine-tuning (e.g. ROUGE-L with REINFORCE)
python3 scripts/summarization.py --lr 0.000003 --fine_tuning --save_dir="multinews_finetune" --model_path='longformer-encdec-large-16384' --from_pretrained='multinews/test/{name_of_last_ckpt}.ckpt' --grad_ckpt --max_output_len 256 --dataset='multinews' --num_dataset_examples='1000' --epochs 1 --custom_method=rouge_reinforce
```

## Acknowledgements
This code is based off of the [Longformer](https://github.com/allenai/longformer) repo.
We also utilise parts of the code from [RELAX](https://github.com/duvenaud/relax), and [Newsroom](https://github.com/lil-lab/newsroom) to implement the coverage reward.