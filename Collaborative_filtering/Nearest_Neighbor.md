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
----------------------
사용자에게 필터링을 최적화 시킨다. 여기서 포인트는 **개인이 아직 평점을 매기지 않은 아이템을 추천**하는 것이다.

이를 위해서 예측 평점의 계산이 필요하다. 예측 평점을 계산하는 식은 다음과 같다.

