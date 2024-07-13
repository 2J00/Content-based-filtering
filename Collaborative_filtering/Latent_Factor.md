# Latent_Factor_Collaborative_filtering
----------------
### 잠재 요인 협업 필터링
사용자 - 아이템 평점 매트릭스 속 **잠재 요인**을 추출해 추천 예측하는 기법 -> 넷플릭스에서 사용하는 추천 기법이다.

잠재 요인이 어떤 것인지 명확히 정의할 수는 없다.

**행렬 분해**를 사용하여 사용자-아이템 평점 행렬을 저차원의 사용자-잠재요인 행렬, 잠재요인-아이템 행렬로 분해한다. 분해된 두 행렬의 내적을 통해 예측 평점 행렬을 생성하고 평점이 부여되지 않은 아이템에 대해 예측 평점을 생성한다.

--------------------
### 실습
MovieLens 데이터 세트를 이용하여 실습을 진행했다.
* 데이터 출처: <https://grouplens.org/datasets/movielens/latest/>

사용자-아이템 행렬에는 Null값이 많기 때문에 Null값이 없는 행렬에만 적용할 수 있는 SVD 행렬 분해는 사용할 수 없다. 따라서 SGD(확률적 경사 하강법)로 행렬 분해를 진행하였다.

----------------------------
먼저 사용자-아이템 평점 행렬을 생성했다.

```
rating_movies = pd.merge(ratings, movies, on = 'movieId')
rating_movies = rating_movies[['userId', 'title', 'rating']]
ratings_matrix = rating_movies.pivot_table('rating', index = 'userId', columns = 'title')
```
-------------------------
다음으로 행렬 분해를 진행했다. 절차는 다음과 같다.
1. P와 Q를 임의의 값을 가진 행렬로 설정한다.
2. P와 Q.T의 값을 곱해 예측 행렬을 계산하고 실제 행렬과의 차이(오류)를 계산한다.
3. 오류 값을 최소화 할 수 있도록 P, Q 값을 업데이트 한다.
4. 만족할 값이 나올 때까지 2, 3의 작업 반복

P와 Q를 업데이트 하기 위해 L2규제가 반영된 비용함수를 적용한다. 비용함수 식은 다음과 같다.

- $\acute{p_u} = p_u + \eta(e_{(u,i)} * q_i - \lambda * p_u)$
- $\acute{q_i} = q_i + \eta(e_{(u,i)} * p_u - \lambda * q_i)$
---------------------------

먼저 오차를 구하는 함수인 get_rmse()를 정의했다. 실제 행렬에서 Null이 아닌 값의 위치 인덱스를 추출해 실제 행렬의 값과 예측 행렬의 동일 위치 값의 RMSE값을 반환한다.

```
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0

    # 예측 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # null이 아닌 값의 위치 인덱스 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    
    # 실제 값 - 예측 값
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse
```
---------------------------------
SGD를 이용한 행렬 분해 함수를 정의한다. 함수의 각 파라미터 값은 다음과 같다.
  * R: 원본 행렬
  * K: 잠재 요인 차원
  * steps: SGD 반복 횟수
  * learning_rate: 학습률
  * r_lambda: L2 규제 계수

```
def matrix_factorization(R, K, steps = 200, learning_rate = 0.01, r_lambda = 0.01):
    num_users, num_items = R.shape
    
    # P와 Q 매트릭스의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale = 1./K, size = (num_users, K))
    Q = np.random.normal(scale = 1./K, size = (num_items, K))

    # R > 0인 행 위치, 열 위치, 값 -> non_zeros 리스트에 저장
    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0]

    # SGD -> P, Q 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제값 - 예측값
            eij = r - np.dot(P[i, :], Q[j, :].T)
            
            # L2 규제 반영한 비용함수 적용 -> P, Q 업데이트
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])
        
        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0:
            print('### iteration step:', step, 'rmse:', rmse)
    
    return P, Q
```
----------------------
정의한 함수를 사용해서 행렬 분해를 진행하고 예측 사용자-아이템 평점 행렬 정보를 만들어 반환하게 했다.

```
P, Q = matrix_factorization(ratings_matrix.values, K = 50, steps = 200, learning_rate = 0.01, r_lambda = 0.01)
```
```
pred_matrix = np.dot(P, Q.T)
ratings_pred_matrix = pd.DataFrame(pred_matrix, index = ratings_matrix.index, columns = ratings_matrix.columns)
```
----------------------
만들어진 예측 사용자-아이템 평점 행렬을 사용해 개인화된 영화 추천을 진행했다. 최근접 이웃 협업 필터링에서 사용했던 함수를 사용했다. 9번 사용자에 대해 영화를 추천하는 코드이다.

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
unseen_list = unseen(ratings_matrix, 9)
recomm_movies = recom_movie(ratings_pred_matrix, 9, unseen_list, n = 10)
recomm_movies = pd.DataFrame(recomm_movies.values, index = recomm_movies.index, columns = ['pred_score'])
recomm_movies
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
      <th>Rear Window (1954)</th>
      <td>5.704612</td>
    </tr>
    <tr>
      <th>South Park: Bigger, Longer and Uncut (1999)</th>
      <td>5.451100</td>
    </tr>
    <tr>
      <th>Rounders (1998)</th>
      <td>5.298393</td>
    </tr>
    <tr>
      <th>Blade Runner (1982)</th>
      <td>5.244951</td>
    </tr>
    <tr>
      <th>Roger &amp; Me (1989)</th>
      <td>5.191962</td>
    </tr>
    <tr>
      <th>Gattaca (1997)</th>
      <td>5.183179</td>
    </tr>
    <tr>
      <th>Ben-Hur (1959)</th>
      <td>5.130463</td>
    </tr>
    <tr>
      <th>Rosencrantz and Guildenstern Are Dead (1990)</th>
      <td>5.087375</td>
    </tr>
    <tr>
      <th>Big Lebowski, The (1998)</th>
      <td>5.038690</td>
    </tr>
    <tr>
      <th>Star Wars: Episode V - The Empire Strikes Back (1980)</th>
      <td>4.989601</td>
    </tr>
  </tbody>
</table>
</div>
