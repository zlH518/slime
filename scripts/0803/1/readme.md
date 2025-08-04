修改了2中的trace的问题，看到现在是可以正常的trace和运行了，接下来首先解决一下乱码的问题，debug查看一下
这里可以确定乱码的问题出现在rollout model offload下去再onload上来就会出现乱码的问题，把这部分注释掉就没有问题了
另外actor model offload下去mem确实减少了2G, 对于tp2来说，确实是对的

4. 测试一个qwen2.5 7B的model，参照example脚本中的配置看一下完整的运行时间，看4个step就可以了
这部分测试出来，wandb中只有1个点，这明显不对劲

6. zk修改完了sglang的onload和offload逻辑，测试一下，现在不会乱码
测试完了，发现actor model真正的offload onload前后没有去打点统计，导致timeline图上面缺了点

7. 测试6补完点之后，并且用0.6b先测试一下
发现这里timeline中ran0-3上面没有offload和onload的显示，添加了一下

8. 测试一下7添加之后的，另外0.6b的模型卸载的太少了，换成4b的试一下
看了一眼其实没有太大的差别，offload前后分别是86G->56G,这个还完全无法接收，又发现在注册mem pool的时候tag应该要注册成带task_id的，修改完了下面继续测试一下

9. 测试一下8里面说的内容
依然不对，完美没有进行update,只能debug查看

10. debug使用0.6b的模型，然后1个任务先看看有没有update步骤


1.继续debug查看update的细节
测试了一下单个任务，发现流程非常的正常，再多测试一下大一点的模型

2.接着1做，测试一下单个任务运行时，多卡的情况
多卡不做tp也没有问题

3.测试多卡做tp，并且用7b的模型做tp 4
timeline来看也没有问题，wandb上面看reward和loss都是0，因为response length我设置的太小了，这是正常的

4，debug一下两个任务的timeline，到底update为什么不释放锁
修改了代码，首先不同的任务创建的通信组需要隔离开，这部分做了修改，但是运行完之后依然卡在那里，测试一下这种情况下不offload有没有问题

5.测试一下不offload
报错了，报了第二个任务的http的进程死亡了，查看打点也是打在第二个任务的init B之前，估计是端口冲突了，打印了一下engine各个初始化的ip和port，第二次运行又不报错了，这部分可能欠缺一点稳定性
task-0: [{'port': 10199, 'nccl_port': 10209, 'dist_init_addr': '172.16.243.139:10279'}, {'port': 10219, 'nccl_port': 10229, 'dist_init_addr': '172.16.243.139:10286'}, {'port': 10239, 'nccl_port': 10249, 'dist_init_addr': '172.16.243.139:10293'}, {'port': 10259, 'nccl_port': 10269, 'dist_init_addr': '172.16.243.139:10300'}]
task-1: [{'port': 11000, 'nccl_port': 11010, 'dist_init_addr': '172.16.243.139:11080'}, {'port': 11020, 'nccl_port': 11030, 'dist_init_addr': '172.16.243.139:11087'}, {'port': 11040, 'nccl_port': 11050, 'dist_init_addr': '172.16.243.139:11094'}, {'port': 11060, 'nccl_port': 11070, 'dist_init_addr': '172.16.243.139:11101'}]
timeline上面看起来没有问题

6.测试一下offload的情况
可以正常运行的，但是发现了一种情况
![alt text](image.png)
![alt text](image-1.png)
就是这里发生了这样一种情况，task1 update完了，然后task0 train完了，之后两个任务都需要同时让sglang onload，并且他们之间再阶段
上面是没有问题的，task1拿rollout的锁，task0拿update的锁，这样就导致了4个卡上面，有的sgalng拉起来是task0的，有的sgalng拉起来是1的，就出现了死锁，但是不明白这里为什么一个机器上面只能拉起来一个引擎吗？(不应该啊，实际使用的时候其实是会出现一个机器上面同时拉
好几个sgalng引擎的情况，这里卡住的原因有待debug分析), 但是即使能够拉起来，也不符合我们任务之间互不影响的目标，因此考虑将三个阶段修改成两个阶段

7.实现一下这里修改成两个阶段，由于考虑到rollout和update都需要先拉起rollout, 所以这里直接将update与rollout阶段合并，这样就不会影响多个阶段直接有相同的model 被onload的情况，并且这样在task init的时候也不需要拉起来再同步参数了，直接offload下去就可以



8.3

1.发现其实只有用了colocate才会用内存池来管理，所以直接把这部分设置成内存池测试一下