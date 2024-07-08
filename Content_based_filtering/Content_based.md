# Content_based_filtering
----------------
### 콘텐츠 기반 필터링

사용자의 선호도가 높았던 아이템과 비슷한 아이템을 추천하는 방식

-----------------

### 실습

* 데이터 출처: <https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>

TMDB 5000 영화 데이터 세트를 가지고 콘텐츠 기반 필터링을 진행하였다.

콘텐츠간 유사성을 판단하는 기준은 다양할 수 있는데, 이번 실습에서는 장르를 기반으로 진행했다. 

--------------------

먼저 필터링의 기준이 되는 genres를 수정한다. genres는 list[dict1, dict2] 형태로 되어있다.

장르의 이름만 리스트에 저장되게 만든다.

```
from ast import literal_eval

mv['genres'] = mv['genres'].apply(literal_eval)
mv['genres'] = mv['genres'].apply(lambda x: [y['name'] for y in x])
```
---------
다음으로 장르간 유사도를 측정한다. genres의 리스트를 문자열로 수정하고, Count를 기반으로 피처 벡터화 변환한다.

```
mv['str_genres'] = mv['genres'].apply(lambda x: (' ').join(x))
```

```
from sklearn.feature_extraction.text import CountVectorizer

cnt_vec = CountVectorizer(min_df = 0.0, ngram_range = (1, 2))
genre_matrix = cnt_vec.fit_transform(mv['str_genres'])
```
-------
CounterVectorizer를 처음 사용해봐서 구조에 대해 찾아보았다.

* 출처: <https://taptorestart.tistory.com/entry/sklearn-textCountVectorizer%EC%97%90%EC%84%9C-ngramrange-%EC%9D%98%EB%AF%B8%EB%A5%BC-%ED%8C%8C%EC%95%85%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94-%EC%98%88%EC%A0%9C>

min_df: 최소 빈도수

ngram_range(min_n, max_n): 단어 몇개를 토큰화 할 것인지를 의미

```
fruit = ['사과 딸기', '딸기 바나나', '수박', '수박 수박']
cv2 = CountVectorizer(min_df=0.0, ngram_range=(1,2))
fv2 = cv2.fit_transform(fruit)
ngram_range = (1,2)


            딸기   딸기 바나나   바나나   사과   사과 딸기   수박   수박 수박
사과 딸기     1         0          0      1         1       0        0
딸기 바나나   1         1          1      0         0       0        0
수박         0         0          0      0         0       1        0
수박 수박     0         0          0      0         0       2        1
```

-----------
유사도 계산에는 코사인 유사도를 사용한다.

유사도 값이 높은 순으로 정렬된 인덱스 값을 추출한다. argsort() 함수를 사용하면 배열을 정렬하는 인덱스 배열을 반환한다.

```
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(genre_matrix, genre_matrix)
ind = sim.argsort()[:,::-1]
```

-----------
이제 장르 유사도를 바탕으로 영화를 추천할 함수를 만든다. 함수 실행시 추천 영화 정보를 가진 데이터 프레임이 반환된다.

추천 영화를 찾을 기반 데이터인 movie_df에서 추천 기준이 되는 영화인 movie_title의 정보를 추출하고, 인덱스 정보를 title_idx에 저장한다.

코사인 유사도 인덱스 리스트인 idx에서 유사도 순으로 top_n개의 인덱스를 추출하여 sim_idxs에 저장한다(2차원 데이터 -> reshape -> 1차원 array로 변경).

기반데이터에서 sim_idxs에 해당하는 정보를 추출하여 반환한다.

```
def find_movie(movie_df, idx, movie_title, top_n = 10):
    title = movie_df[movie_df['title'] == movie_title]
    
    title_idx = title.index.values
    sim_idxs = idx[title_idx, :(top_n)]

    sim_idxs = sim_idxs.reshape(-1)

    return movie_df.iloc[sim_idxs]
```

함수를 사용하여 영화 '대부(The Godfather')와 장르별로 유사한 영화 5편을 추천해보았다.

```
movies = find_movie(mv, ind, 'The Godfather', 5)
movies[['title', 'vote_average']]
```
