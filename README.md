# 图片全景合成作业结果
本次实验为图像拼接，调用opencv提取图像的特征值，并进行匹配。自己写了 RANSAC算法对特征值筛选。
### 实验结果
#### 原始图片
 <img src="https://i.loli.net/2019/11/14/RTlpOxfKrzbm45D.jpg" width = "200" />
 <img src="https://i.loli.net/2019/11/14/tyKwjORETr5cALF.jpg" width = "200" />
#### 拼图结果
 <img src="https://i.loli.net/2019/11/14/ks23QItv6OJoNdh.png"  />


#### 具体步骤
1. 调用opencv计算两张图片的特征值
```python
img1 = cv2.imread('picture/1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('picture/2.jpg', cv2.IMREAD_COLOR)
sift = cv2.xfeatures2d.SIFT_create()
kp1, descriptors1 = sift.detectAndCompute(img1, None)
kp2, descriptors2 = sift.detectAndCompute(img2, None)
```

2. 调用opencv对两幅图的特征值进行匹配
```python
def match_point(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return good

matches = match_point(descriptors1, descriptors2)
```

匹配后的结果为
```python
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
						  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

<img src="https://i.loli.net/2019/11/14/WnQcbqJsDPLlF4X.png"  />

3. 由于很多点对匹配不好，自己写RANSAC算法进行筛选
```python
def ransc(kp1, kp2, t, deviation):
    kp1_add_one = np.hstack((kp1, np.ones((kp1.shape[0], 1))))
    index = range(len(kp1))
    max_correct, ans_index = 0, []
    while t:
        t -= 1
        selected = random.sample(index, 4)
        homo, _ = cv2.findHomography(kp1[selected], kp2[selected])

        dst = np.dot(kp1_add_one, homo.T)
        dst = dst/(dst[:, -1].reshape(dst.shape[0], 1))
        dst = dst[:, :2]

        correct, ans = 0, []
        for i in index:
            if np.linalg.norm(dst[i] - kp2[i]) < deviation:
                correct += 1
                ans.append(i)

        if max_correct <= correct:
            max_correct = correct
            ans_index = ans

    return ans_index

fir_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1,  2)
sec_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1,  2)
sec_pts[:, 0] += w
index = ransc(fir_pts, sec_pts, 100, 10)
```

筛选完后结果明显好多了
```python
img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, np.array(matches)[index], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
<img src="https://i.loli.net/2019/11/14/DYUOBohMmvQ6zx1.png" />

4. 最后对第一张图片转化视角，并且进行拼接
```python
homography, _ = cv2.findHomography(fir_pts[index], sec_pts[index])
img5 = cv2.warpPerspective(img1, homography, (w*2, h))
img5[0:h, w:w*2] = img2
plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
<img src="https://i.loli.net/2019/11/14/VrTAGh7xt6Fjo9k.png"  />
