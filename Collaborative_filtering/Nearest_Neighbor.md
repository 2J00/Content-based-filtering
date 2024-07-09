# Nearest_Neighbor_Collaborative_filtering
--------------------
### 최근접 이웃 협업 필터링

취향이 비슷한 사용자의 행동양식을 기반으로 추천 -> 의류 쇼핑몰에서의 '다른 고객이 함께 찾은 상품' 이나 유튜브 알고리즘의 '시청자가 이 동영상도 시청함' 등이 해당한다고 보면 될 것 같다.

사용자 행동 데이터(사용자-아이템 평점 매트릭스) -> 사용자가 평가하지 않은 아이템을 예측 평가

최근접 이웃 협업 필터링은 사용자 기반과 아이템 기반으로 나뉠 수 있다.

  * 사용자 기반: 당신과 비슷한 고객들이 구매 -> 특정 사용자와 타 사용자 간 유사도를 측정하여 유사도가 높은 N명의 사용자를 추출해 그들이 선호하는 아이템을 추천

  * 아이템 기반: 이 상품을 선택한 다른 고객들이 구매 -> 아이템 선호도 평가가 유사한 아이템을 추천

일반적으로 아이템 기반 협업 필터링이 정확도가 더 높다. 따라서 대부분 최근접 이웃 협업 필터링은 아이템 기반 필터링을 적용한다.

-------------------
### 실습

MovieLens 데이터 세트를 이용하여 아이템 기반 협업 필터링을 진행했다. 유사도 측정에는 코사인 유사도를 사용했다.

* 데이터 출처: <https://grouplens.org/datasets/movielens/latest/>
----------------------
먼저 최근접 이웃 협업 필터링의 사용자 행동 데이터인 사용자-아이템 평점 매트릭스를 만든다. 사용자가 평점을 매기지 않아 생긴 null값은 0으로 대체했다. 가독성을 위해 아이템 정보인 movieId를 title로 변경하였다.

```
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_mat = ratings.pivot_table('rating', index = 'movieId', columns = 'userId')

rating_movies = pd.merge(ratings, movies, on = 'movieId')
ratings_mat = rating_movies.pivot_table('rating', index = 'title', columns = 'userId')

ratings_mat = ratings_mat.fillna(0)
```
---------------------
코사인 유사도를 사용해 영화 평점이 얼마나 비슷한지 유사도를 산출한다. 

cosine_similarity() 함수는 행을 기준으로 서로 다른 행을 비교한다는 점을 유의해야 한다. 행렬의 전치가 필요할 경우 transpose() 함수를 사용하면 된다.

```
item_sim = cosine_similarity(ratings_mat, ratings_mat)
item_df = pd.DataFrame(data = item_sim, index = ratings_mat.index, columns = ratings_mat.index)
```

영화 '대부'와 유사도가 높은 영화를 추출해보았다. 자신을 제외하고 10위까지의 유사한 영화가 출력되도록 했다.

```
item_df['Godfather, The (1972)'].sort_values(ascending = False)[1:11]
```
```
# 결과
title
Godfather: Part II, The (1974)                           0.821773
Goodfellas (1990)                                        0.664841
One Flew Over the Cuckoo's Nest (1975)                   0.620536
Star Wars: Episode IV - A New Hope (1977)                0.595317
Fargo (1996)                                             0.588614
Star Wars: Episode V - The Empire Strikes Back (1980)    0.586030
Fight Club (1999)                                        0.581279
Reservoir Dogs (1992)                                    0.579059
Pulp Fiction (1994)                                      0.575270
American Beauty (1999)                                   0.575012
Name: Godfather, The (1972), dtype: float64
```
----------------------
사용자에게 필터링을 최적화 시킨다. 여기서 포인트는 **개인이 아직 평점을 매기지 않은 아이템을 추천**하는 것이다.

이를 위해서 예측 평점의 계산이 필요하다. 예측 평점을 계산하는 식은 다음과 같다.

![예측평점 계산](https://github.com/2J00/Filtering_/blob/main/Collaborative_filtering/%EC%98%88%EC%B8%A1%20%ED%8F%89%EC%A0%90.png)

 * u: 사용자, i: 아이템
 * S: 아이템 i와 가장 유사도가 높은 N개 아이템의 유사도 벡터
 * R: 사용자 u의 아이템 i와 가장 유사도가 높은 N개 아이템에 대한 실제 평점 벡터

다음은 사용자별로 추천을 최적화하기 위한 예측평점 계산 함수이다(N의 범위에 제약 X). 위의 수식과 내용은 같다.

u의 모든 영화에 대한 실제 평점과 i의 다른 모든 영화와의 코사인 유사도를 내적 곱한 값을 정규화 한다.
```
def predict_rating(ratings_arr, item_sim_arr):
    pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis = 1)])
    return pred
```
```
ratings_pred = predict_rating(ratings_mat.transpose().values, item_df.values)
pred_mat = pd.DataFrame(data = ratings_pred, index = ratings_mat.columns, columns = ratings_mat.index)
```
----------------------
예측력을 높이기 위해 특정 영화와 가장 비슷한 유사도를 가지는 영화에 대해서만 유사도 벡터를 적용하는 함수로 변경했다.

```
def pred_rating_top(ratings_arr, item_sim_arr, n):
    pred = np.zeros(ratings_arr.shape)

    for col in range(ratings_arr.shape[1]):
        top_n = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n].dot(ratings_arr[row, :][top_n].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n]))
    
    return pred
```
```
res = pred_rating_top(ratings_mat.transpose().values, item_df.values, 20)
pred_mat = pd.DataFrame(data = res, index = ratings_mat.transpose().index, columns = ratings_mat.transpose().columns)
```
-------------
사용자가 아직 보지 않은 영화들만 추천할 수 있도록 보지 않은 영화의 리스트를 추출하고, 평점을 예측하여 추천하는 함수를 만들었다.

2번 사용자에게 영화를 추천해보았다.

```
def unseen(matrix, userID):
    user_rating = matrix.loc[userID, :]
    already = user_rating[user_rating > 0].index.tolist()

    movie_lst = matrix.columns.tolist()
    unseen_lst = [movie for movie in movie_lst if movie not in already]

    return unseen_lst
```
```
def recom_movie(pred_df, userID, unseen_lst, n):
    recom = pred_df.loc[userID, unseen_lst].sort_values(ascending = False)[:n]
    return recom
```
```
unseen_mv = unseen(ratings_mat.transpose(), 2)
recom = recom_movie(pred_mat, 2, unseen_mv, 10)
recom = pd.DataFrame(data = recom, index = recom.index, columns = ['pred_score'])
```

<div>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_score</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Social Network, The (2010)</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>The Imitation Game (2014)</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Gran Torino (2008)</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Grand Budapest Hotel, The (2014)</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>District 9 (2009)</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
