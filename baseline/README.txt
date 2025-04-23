需要从 https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main 下载文件 pytorch_model.bin 放置在文件夹 chinese_robert_wwm_ext_large中.


用python3先后依次运行训练文件train_task1.py, train_task2.py, train_task3.py得到训练参数，参数会保存在saves文件夹中.

得到参数后先后依次运行B轮的预测文件predict_task1, predict_task2, predict_task3即可得到结果A_task1_test, A_task2_test, A_task3_test于文件夹dataset中.
（注意必须先预测task2任务得到B_task2_test，再预测task3任务！）

提交的结果在文件夹SOTA中.


zip submit.zip A_task1_test.json A_task2_test.json A_task3_test.json


/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/we-falsefreeze:
    task1:nofreeze, task2:使用了777 seed Roberta， task3使用了777seed，task2/3都是最好的结果下（task3是加了ws/target we和focal loss)
    是2025/4/22日的第一次结果

/script: 用于分析数据
/get_dep:用来生成pos/dep数据，但是目前还没有用到
