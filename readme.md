## 1.�Լ����������main idea:
����������Ȼ��ѭkmeans���̣�
![alt text](img\image11.png)
�������Ż������ķ���͸��µ�loss������
![alt text](img\image12.png)

![alt text](img\image.png)
���� q ���ȷֲ��ڵ�λ�����ϵ������
�������е� q �� x �Զ�ͬ����Ҫ������ x ���ԣ�׼ȷ����<q1; x>���ڻ���<q2; x> �� <q3; x> ����Ҫ��
��Ϊ<q1; x>���ڻ�������˸��п��������ֵ��
![alt text](img\image1.png)
![alt text](img\image2.png)
![alt text](img\image3.png)
?q,x?=�O�Oq�O�O?�O�Ox�O�O?cos(��)
���У��� ������ q �� x ֮��ļнǡ�
��:=arccos(�O�Ox�O�Ot?)
����� t ���ڻ� ?q,x? ��ֵ
![alt text](img\image4.png)

<mark>**���յĸ�������loss�� eta ����ƽ�з������ϴ�ֱ����**��

������֪��ʧ������һ��ֱ�ӽ���ǣ�Ŀ�����ݵ�Ե���Ҫ�Խ���Ȩ�⣬�Ӷ����Ͷ�������ǰ����ԵĹ�����

## 2.google��Ŀ����
https://github.com/google-research/google-research/tree/master/scann

anisotropic loss�ĺ���ʵ����scann\partitioning\anisotropic.cc

�ؼ�������eta,��Ҫnormalize dataset,ÿ�����ݵ����һ����etaֵ��

eta����˼·��
![alt text](img\image7.png)
![alt text](img\image8.png)
\scann\hashes\internal\stacked_quantizers.h��
![alt text](img\image9.png)
![alt text](img\image10.png)

![alt text](img\image6.png)
ʵ�����ǵĴ����м���ģ�
![alt text](img\image5.png)
��ReducePartition�����



