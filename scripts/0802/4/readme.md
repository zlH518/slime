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