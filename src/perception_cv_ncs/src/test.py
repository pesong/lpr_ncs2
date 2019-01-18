import tensorflow as tf

x1 = tf.constant([[[3., 3.],[3., 3.]],[[3., 3.], [3., 3.]]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
x2 = tf.constant([[[3., 3.],[3., 3.]],[[3., 3.], [3., 3.]]])

output_list = []


# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
for i in range(2):
    output_list.append(tf.matmul(x1[i,:], x2[i,:]))

outputs = tf.stack(output_list)

# 启动默认图.
with tf.Session() as sess:
    sess.run(outputs)
    print(outputs.eval())

print()
