# turing

#### 介绍
homework recommendation  
- 模型为双向GRU，历史作业取14天，双向补零5天
- 模型BERT，历史作业取14天，双向补零5天

#### 模型日更

    定时任务 模型更新
    crontab -l
    /data/zhangkl/turing_new/crontab_config_online
    30 10 * * * nohup python3 -u /data/zhangkl/turing_new/handle.bgru_online.py >> /data/zhangkl/turing_new/online.log 2>&1 &
    

#### 上线部署
    机器
    ubuntu@10.19.x.x:/data/turing/turing
    ubuntu@10.19.x.x:/data/turing/turing
    
    定时任务 模型只保留最新五个
    crontab -l
    30 15 * * * nohup python -u /data/turing/timer_del_model.py >> /data/turing/timer_del_model.log 2>&1 &
