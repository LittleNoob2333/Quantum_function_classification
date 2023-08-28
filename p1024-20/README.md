

# 安装依赖

## 下载 项目实例代码
```
git clone http://gitlab.turingq.com/zhaoxiang/challenge
```


## 安装 matplotlib torch

```
conda create -n challenge python=3.10
conda activate challenge
```

## 安装 deepquantum 和其他依赖
解压 `deepquantum-v0.0.4.zip`
```
cd deepquantum-v0.0.4
pip install .
```

```
cd p1
pip install -r requirements.txt
```


# 运行示例代码

```
cd p1/
```

## 算法一：Deutsch-Jozsa 算法
```
python main.py dj
```


## 算法二：PQC 算法
```
python main.py pqc
```