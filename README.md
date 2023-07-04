# AI_Project
I have covered 5 major disasters:
1. Tsunami
2. Wildfire
3. Torando
4. Earthquake
5. Hurricanes
The following are the links to my ML algorithms and analysis used :

https://colab.research.google.com/drive/1vsVlH4YaePSTa6pWOyzwN_5NdwMsQWHm?usp=sharing (Earthquake)
https://colab.research.google.com/drive/1ctpWiHNGmwT-cs2fSrEzlQHCe6MbLI6_?usp=sharing (Hurricanes)
https://colab.research.google.com/drive/19YCecDDTwbJPaCflr2zHIQxFwVkyDzKb?usp=sharing (Wildfire)
https://colab.research.google.com/drive/1SBMK2jBghvQjqVfr5eRN38l_o2CZg_z7?usp=sharing (Tsunami)
https://colab.research.google.com/drive/1b8xHZ9xk-kMr6g8BHpP9BdzNRXJxa_Ax?usp=sharing (Tornado)

The following are the links to databases used:
https://www.kaggle.com/datasets/jahaidulislam/significant-earthquake-dataset-1900-2023 ( Eathquake)
https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021 (Torando)
https://www.kaggle.com/datasets/andrewmvd/tsunami-dataset (Tsunami)
https://www.kaggle.com/datasets/noaa/hurricane-database (Hurricane)
https://www.kaggle.com/datasets/mbharti321/algerian-forest-fires-dataset-updatecsv (Wildfire)

Code snippets showing analysis done on the database:
https://colab.research.google.com/drive/1b8xHZ9xk-kMr6g8BHpP9BdzNRXJxa_Ax?authuser=1#scrollTo=hmDi708fNTJs&line=12&uniqifier=1

Comparison of multiple models( approaches):
```
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x3, y3)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df
```

My research and analysis shows that:
a. Wildfires depend upon several factors like: Temperature,	RH,	Ws,	Rain,	FFMC,	DMC,	DC,	ISI,	BUI,	FWI ( last 5 represent various environmental indices)
b. Earthquake depends upon : Latitude,	Longitude,	Depth, nst,	gap,	dmin,	rms, magNst
c. Hurricanes depend on : location (atlantic/pacific), maximum wind and minimum pressure
d. Tsunami depends on: magnitude	cdi,	mmi,	alert,	sig,	net,	nst,	dmin,	gap,	magType,	depth,	latitude,	longitude
e. Tornado depends on: st,	mag,	inj,	fat,	slat,	slon,	elat,	elon,	len,	wind
![Alt heatmap] (.heatmap)
![Alt confusion matrix] (.confusion_matrix)
