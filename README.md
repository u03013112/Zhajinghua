# Zhajinghua

调试方式：
浏览器打开 http://u03013112.win/zhajinhua/index.html
web端游戏是websocket客户端，可以接受reset，step等请求并作出回应，具体请参照Step1.py
Env.py使用了蹩脚的阻塞方式使得网络发送可以阻塞直至网络回应，使AI代码更容易写。
先运行Step1.py，等待提示“Listening on port 9001 for clients..”后网页上点击连接，即开始第一步数据采集工作。

游戏规则，先reset-》随机发5张牌。然后动作0，1，2，3，4 这5个动作分别代表将第n张牌拿出摆在摆牌区。
摆满三张开始算分，规则如下：
豹子：  100分
同花顺：80分
同花： 60分
顺子： 40分
对子：20分
散牌：0分

顺随机大致会得到7~8分左右，目前我的成绩是25左右，争取能达到30以上。

如有建议请留言~
