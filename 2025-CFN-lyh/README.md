需要从 https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main 下载文件 pytorch_model.bin 放置在文件夹 chinese_robert_wwm_ext_large中.

dataset/cfn_ws：是我处理之后的数据，加了分词信息和词性信息
cc.zh.300.cfn.vec
cc.zh.300.cfnAB.vec 是我抽取的fasttext中的预训练数据
##### 训练部分 ######
```
cd 2025-CFN-lyh

用python3先后依次运行训练文件train_task1.py, train_task2.py, train_task3.py得到训练参数，参数会保存在saves文件夹中.
nohup python -u train_task1.py > log/train-task1.log 2>&1 &
nohup python -u train_task2.py > log/train-task2.log 2>&1 &
nohup python -u train_task3.py > log/train-task3.log 2>&1 &

notes:注意更改GPU编号：os.environ['CUDA_VISIBLE_DEVICES']
```
##### 测试部分 ######

* 任务1:分两步，先跑出三个预测结果，然后投票
(1) nohup python -u predict_task1.py > log/predict-task1.log 2>&1 &
(2) cd get_dep
    python vote_task1.py  

* 任务2:
  （1） nohup python -u predict_task2.py > log/predict-task2.log 2>&1 &
得到参数后直接运行B轮的预测文件 predict_task2就可以得到结果A_task2_test于文件夹dataset中
* 任务3:分两步，先跑出三个预测结果，然后投票（注意必须先预测task2任务得到B_task2_test，再预测task3任务！）
(1） nohup python -u predict_task3.py > log/predict-task3.log 2>&1 &
(2)  cd get_dep
     python vote_task3.py 

先跑出三个预测结果，然后投票


即可得到结果B_task1_test, B_task2_test, B_task3_test于文件夹submit中.

