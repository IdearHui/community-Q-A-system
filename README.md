# 基于社区的问答系统设计与实现
## 一、任务简介
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;社区问答系统（如知乎、Quora 等）越来越受到互联网用户的关注。由于其开放的特点，任何用户都可以提出问题或者是回答其他用户提出的问题，因此一个问题具有几百甚至上千的答案也是不足为奇的。但是答案的质量却参差不齐，和问题的相关程度也千差万别，这使得用户要把所有的答案阅读一遍成为一件费时费力的事情。针对这种情况，我们提出下面两个子任务：任务1 的目的是自动识别相关有用的答案，任务2 针对一般疑问句问题（即答案为Yes/No 的问题），从所有答案中总结出问题的正确答案。下面是任务的详细介绍。
## 二、任务题目
### (一)	任务一
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给定训练数据为若干问题（Question）以及每个问题对应的若干答案（Comment），答案被标记为下面几种标签中的任意一种（CGOLD）：   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a.	Good：答案和问题相关；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.	Bad：答案和问题不相关；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.	Potential：潜在有用的答案；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d.	Dialogue：答案为用户之间的对话，并非正面回答问题；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e. 	Not English: 非英语；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f.	Other：其他。   
测试时给定问题和答案，对每个答案的类别进行预测。
### (二)	任务二
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给定训练数据为一般疑问句问题以及每个问题对应的若干答案，答案被标记为下面几种标签中的任意一种（CGOLD_YN）：   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a.	Yes：肯定回答；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.	No：否定回答；   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.	Unsure：不确定。   
测试时给定形式为一般疑问句的问题及若干答案，综合考虑每个答案的描述，对每个问题的答案（Yes/No/Unsure）进行预测。
## 三、实验过程
### (一)	实验预处理
#### a.	去掉停用词和标点符号  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;有一些功能词语像定冠词the不定冠词a, an 以及一些介词 about, above, across, after等，这些功能词极其普遍，与其他词相比，功能词没有什么实际含义，当所要搜索的短语或文本包含功能词，则难以帮助缩小搜索范围，同时还会降低搜索的效率，所以通常会把这些词从问题中移去，从而提高搜索性能。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在任务二中，由于要判断yes、no以及unsure等概率性的结果，因此任务二的停用词中应去掉表达揣测、肯定、否定等语气的词语，比如“yes”“no”“think”等。根据对训练数据的统计，要在任务二的停用词中去掉如下词语："no", "dont", "n't""yes", "sure", "almost", "just""think", "what"  
#### b.	序列化  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这里所说的序列化其实是用了Python的pickle模块，它是一个可以将对象数据持久化的一个技术，可以将对象 pickle 成字符串、磁盘上的文件或者任何类似于文件的对象（使用dump()方法），也可以将这些字符串、文件或任何类似于文件的对象 unpickle 成原来的对象(使用load()方法)。这样可以将源数据文件（此处指所要用到的xml文件）转化成中间数据文件存储起来，方便结构化以及使用。  
### (二)	模型方法
#### 1.	TF-IDF (term frequency–inverse document frequency)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。TF-IDF加权的各种形式常被搜索引擎应用，作为文件与用户查询之间相关程度的度量或评级。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其主要思想是如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
#### 2.	LSA (Latent Semantic Analysis)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LSA以向量空间模型作为基础，将文档表示映射到潜在语义空间，从而更好地衡量文本之间的相关性。LSA也常被视作一种维数缩减技术，因为它把文档从高维的词项空间映射到低维潜在语义空间，去除了噪音。其基本思想是基于词项在文档集合中的共现特性，表达词项的潜在意义。
#### 3.	SVM (Support Vector Machine)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SVM是一个有监督的学习模型，通常用来进行模式识别、分类、以及回归分析。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其主要思想是针对线性可分情况进行分析，对于线性不可分的情况，通过使用非线性映射算法将低维输入空间线性不可分的样本转化为高维特征空间使其线性可分，从而使得高维特征空间采用线性算法对样本的非线性特征进行线性分析成为可能。
### (三)	所用特征
#### 1.	任务一
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;任务一中，通过LSI提取了225个主题特征值，另外，加上三个附加特征:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a.	是否包含问号  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这里我们做了一个假设，一般一个评论中，都是肯定回答，若包含问句，那么这个评论是dialog的可能性更高。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.	QUSERID == CUSERID   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们也可以通过评论的用户是否和该评论的提问者是否是同一人做抉择，如果是同一个人评论自己的回答，那么该条评论是dialog的可能性也很大。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.	词数判断  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们根据对开发数据（dev.xml）的统计分析得出了一个粗略的结果，即几乎good的评论的单词数都达到了50个以上，因此我们将词数大于50也作为一个特征，用于标记good的回答。  
#### 2.	任务二
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在任务一的基础上增加了三种类型的附加特征：positives，negatives和unsure三类，分别包含了:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;positives = {"yes", "sure", "almost", "just"}  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;negatives = {"no", "dont", "n't"}  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;unsure = {"think", "what"}  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;即包含positive词语的评论会更可能为yes，包含有negative词语的更可能为no，包含unsure词语的更可能为unsure。以此来提高模型的准确率。
### (四)	工具及资源
#### 1.	Sklearn (scikit-learn)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sklearn是Python中一个非常强大的数据分析工具包，这里我们使用sklearn的支持向量机SVM(其中的支持向量分类器SVC)做数据分类。
#### 2.	NumPy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NumPy系统是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多（该结构也可以用来表示矩阵（matrix））。
#### 3.	SciPy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SciPy是一款方便、易于使用、专为科学和工程设计的Python工具包。它包括统计，优化，整合，线性代数模块，傅里叶变换，信号和图像处理,常微分方程求解器等等。
#### 4.	Gensim
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gensim是一款免费的Python工具包，是一种可伸缩的统计语义，可用来分析纯文本的语义结构，检索语义相似的文件。
### (五)	实验运行及结果分析
#### 任务一：
##### 1.	运行过程及测试
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于数据不同，对于开发数据（针对于dev.xml）和测试数据（针对于test.xml）以及任务一和任务二，我们需要在Main.py文件中更改一些参数运行程序。 同时修改测试数据或是开发数据路径。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;针对具体的情况进行设置后，运行Main.py 文件即可运行程序。准确率是由经过模型预测的结果与dev-gold.txt文件结果进行对比得出。
##### 2.	实验结果：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上过程为利用train.xml 作为训练数据，利用dev.xml 数据进行开发测试的过程，现在我们将test.xml 文件作为测试数据，依旧使用train.xml 作为训练数据，运行过程与上述步骤一样。
##### 3.	结果分析：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;生成的test_gold.txt和test_gold-yn.txt 文档中都有两列数据，test_gold.txt即任务一结果中左边一列数据为测试数据中答案的评论ID（即CID）右边一列为预测的答案评价。test_gold-yn.txt即任务二结果中左边为问题ID（即QID）右边为预测的答案结果（yes，no或者unsure）
### (六)	实验结论
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;经过实验测试，针对开发数据，任务一的准确率可以达到63%左右，任务二的准确率可以达到50%左右。但是在测试的过程中，通过不断的调整参数以及附加特征，会使得实验的效果发生变化，寻找更多的特征会使得判断或预测的准确率更高。同时本次项目中使用的分类模型是sklearn SVM，有些不太稳定，计算出来的准确率会有浮动，可以尝试采用更好的模型，比如台湾大学林智仁教授等开发设计的LIBSVM工具等。同时在任务二中，对于参数及特征的选择考虑的还不是很周全，仍然有许多可改进之处，在最后问题答案的判断时，可以考虑从第一问的基础上，进行预测判断（比如给good的comment赋予更高的权值，其评论的CGOLD_YN可以着重考虑）等。
## 四、	参考文献
http://www.cnblogs.com/lifegoesonitself/p/3506886.html  
http://scikit-learn.org/stable/modules/svm.html#svc  
http://blog.csdn.net/roger__wong/article/details/41175967  
