# OAG Entity Alignment Task

## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/zfjsail/OAG-entity-alignment.git
cd OAG-entity-alignment
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1n9kq3d-mJ0y1k7K4Bei-9g) (with password 445n) or [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/entity-matching/el_data.zip). Unzip the file and put the _data_ directory into project directory.

## How to run
```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
python processing.py  # all pre-processing process


python train_rnn_match.py --n-try 5 --entity-type author  # RNN-based matching model for author alignment
```

If you want to know about all supported baselines, see ```run.sh``` for details.
For entity types, we support three types of entities (*affliation*, *venue*, and *author*) for entity alignment.
For matching models, we support *RNN-based*, *CNN-based*, and *heterogeneous graph attention network (HGAT) based* models.

## References
🌟 If you find our work helpful, please leave us a star and cite our paper.
```
@inproceedings{zhang2024oag,
  title={OAG-bench: a human-curated benchmark for academic graph mining},
  author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6214--6225},
  year={2024}
}
@article{zhang2022oag,
  title={Oag: Linking entities across large-scale heterogeneous knowledge graphs},
  author={Zhang, Fanjin and Liu, Xiao and Tang, Jie and Dong, Yuxiao and Yao, Peiran and Zhang, Jie and Gu, Xiaotao and Wang, Yan and Kharlamov, Evgeny and Shao, Bin and others},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={35},
  number={9},
  pages={9225--9239},
  year={2022}
}
@inproceedings{zhang2019oag,
  title={OAG: Toward linking large-scale heterogeneous entity graphs},
  author={Zhang, Fanjin and Liu, Xiao and Tang, Jie and Dong, Yuxiao and Yao, Peiran and Zhang, Jie and Gu, Xiaotao and Wang, Yan and Shao, Bin and Li, Rui and others},
  booktitle={Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={2585--2595},
  year={2019}
}
```
