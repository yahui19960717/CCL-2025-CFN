from datetime import datetime
import time

def get_time_diff(start_time, end_time):
    # 计算时间差
    time_diff = end_time - start_time

    # 将时间差转换为秒数
    total_seconds = time_diff.total_seconds()

    # 计算时分秒
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # 输出结果
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"运行时间为: {formatted_time}")
