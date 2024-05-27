# Project Title: Challenge Data 35

This challenge focuses on the topic of large-scale product type code multimodal (text and image) classification where the goal is to predict each product’s type code as defined in the catalog of Rakuten France. 
In this project, I am going to restrict to the text analysis, it can go further in the future if time is provided

## Table of Contents

- [Installation]None
- [Usage]Classification of products by product types
- [Contributing]Hamza ERRAHJ, Quentin FOURNEL
- [License]None

# Rapport projet initiation aux Data sciences

### Productdataclassificationinto producttypecodes

## RakutenFranceMultimodalProductDataClassification

## Participants:

## E.Hamza

## hamza.errahj@student-cs.fr

## F.Quentin

## fournier.quentin69@gmail.com

### liengithub:https://github.com/hamzawillgrow/challdata


## Sommaire:

## 1. Contexte

## 2. Objectif

## 3. Prétraitementdutexte

## 4. ObtentiondelamatriceTF-IDF

## 5. Modélisation

## a. SVM

## b. LogisticReg

## c. KNN

## d. Randomforest

## e. VotingClassifier

## 6. Comparatif

## 7. Conclusion


1. **Contexte**

Cechallengeportesurlaclassificationdesproduits.L'objectifestdeprédirelecodetype
(prdtypecode) de chaque produit en utilisant des données textuelles (désignation et
description)etdesimagesissuesducataloguedeRakutenFrance.

Pournotreprojetnousnetiendronspascomptedesimagesàdesfinsdesimplification.

2. **Objectifduprojet**
- Prétraiterletexte
- ConvertirlesdonnéesdansunematriceTF-IDF
- Appliquerdifférentesapprochessurcesdonnéespourprédirelavariable
- Comparerlesperformances
- Concluresurl’approchelaplusappropriée
3. **Prétraitementdutexte**

Pourcedéfi,RakutenFranceproposeenviron 99000 listesdeproduitsauformatCSV,
incluantàlafoisl'ensembled'entraînement(84916)etl'ensembledetest(13812)pour
lequelonnepossèdepasleslabels.Lejeudedonnéescomprend 4 features:

- Désignationsdeproduits,
- Descriptionsdeproduits,
- Imagesdeproduits,
- Codesdeproduitscorrespondants.

1.“X_train_update.csv”:untableaucontenantdeséchantillonsd'entraînementavecla
descriptiontextuelleetlaréférencedufichierimageassocié.
2.“y_train_CVw08PX.csv”:untableaucontenantles 84916 codesdesproduits.
3.“X_test_update.csv”:untableaucontenantdeséchantillonsdetest.
4.“images.zip”:unfichierquiregroupetouteslesimages,ilneserapasutilisépourcette
partie

Lesfichiers“X”CSVsontorganisésdelamanièresuivante :

- **Id**:unidentifiantentierpourleproduit,utilisépourassocierleproduitàsoncodede
produit.
- **designation**:letitreduproduit,untextecourtrésumantleproduit.
- **description**:untexteplusdétaillédécrivantleproduit.Cechamppeutcontenirdes
valeursmanquantescartouslesutilisateursnelerenseignentpas.
- **productid**:unidentifiantuniquepourleproduit.
- **imageid**:unidentifiantuniquepourl'imageassociéeauproduit.
Deplus,onremaplesvaleursdesclassespoursimplifierlalectureetledéveloppement,
toutendiminuantlesvaleursquisontcalculées(26<<2905)cequipeutaccélererlecalcul.


```
ObtentiondelamatriceTF-IDF
```
```
LaconversiondesdonnéestextuellesenunematriceTF-IDF(TermFrequency-Inverse
DocumentFrequency)estuneétapecrucialepourtransformerlestextesenuneforme
numériqueutilisableparlesalgorithmesdemachinelearning.
```
```
LeTF-IDFestunetechniquequipermetdemesurerl'importanced'untermedansun
documentparrapportàuncorpusdedocuments.Cetteimportanceestcalculéeen
multipliantdeuxmétriques:lafréquenced'untermedansundocument(TF)etl'inversede
lafréquencedutermedanslecorpus(IDF).
```
```
Pourceprojet,nousavonsutilisélabibliothèque scikit-learn pourgénérerlamatrice
TF-IDF.Lesétapessontlessuivantes:
```
```
● Concaténationdeschampstextuels :Pourchaqueproduit,nousavonscombiné
leschamps"designation"et"description"enunseulchamptextuel.
● Nettoyagedestextes :Nousavonsconvertilestextesenminuscules,suppriméles
caractèresspéciaux,lesstopwordseteffectuéunelemmatisationpourréduireles
motsàleurformeracine.
● CalculduTF-IDF :Utilisationde TfidfVectorizer de scikit-learn pourtransformer
lestextesenunematriceTF-IDF.
```
fromsklearn.feature_extraction.text importTfidfVectorizer
_#Concaténation des champs textuels(fonction create_text utilisée dansle git)_
for iinrange(xtrain.shape[ 0 ]):
xtrain['text'][i] =create_text(xtrain['designation'][i],
xtrain['description'][i])

_#Nettoyage des textes (exemple)_
xtrain['text']= xtrain['text'].apply(lambdatext : lower_case(text)) # Conversion
enminuscules
***.apply(lambda text :remove_accent(text))# Suppressiondes accents

***.apply(lambda x:' '.join([word for word inx ifword not instopwords]))

#stopwords
***.apply(lambda text :remove_htmltags(text)) #Suppressiondes encodageshtmls

***.apply(lambda text :keeping_essentiel(text))

_#Calcul duTF-IDF_
tfidf_vectorizer= TfidfVectorizer(max_features= 5000 )
X_tfidf= tfidf_vectorizer.fit_transform(X_train['text'])


5. **Modélisation**

Pourlamodélisation,nousavonsexpérimentéavecplusieursalgorithmesdeclassification
pourprédirelescodesdeproduit(prdtypecode).Lesalgorithmeschoisisincluent:

```
● SupportVectorMachines(SVM)
● RégressionLogistique(LogisticRegression)
● k-NearestNeighbors(KNN)
● RandomForest
● VotingClassifier
```
**SupportVectorMachines(SVM)**

LeSVMestunclassifieurlinéairepuissant,particulièrementefficacedanslesespacesde
grandedimension.Ilfonctionnebienpourlaclassificationdetextesgrâceàsacapacitéà
trouverunhyperplanoptimalquiséparelesclasses.

```
from sklearn.svm import SVC from sklearn.metricsimport accuracy_score
svm = SVC(kernel='linear', C= 1 ) svm.fit(X_tfidf,y_train) y_pred_svm =
svm.predict(X_tfidf_test) accuracy_svm = accuracy_score(y_test,
y_pred_svm)
```
**RégressionLogistique**

Larégressionlogistiqueestunmodèledeclassificationlinéairesimplequipeutêtretrès
efficacepourlaclassificationdetexteenraisondesacapacitéàgérerlesproblèmesde
classificationbinaireetmulticatégorie.

```
from sklearn.linear_model import LogisticRegression logreg =
LogisticRegression(max_iter= 1000 ) logreg.fit(X_tfidf, y_train)
y_pred_logreg = logreg.predict(X_tfidf_test) accuracy_logreg =
accuracy_score(y_test, y_pred_logreg)
```
```
Matricedeconfusion
```

Onremarquesurlamatricedeconfusionquelesclassesquiprésententplusdedonnées
pourl’apprentissageontplusderéussiteàêtredétectéequelesautres(classe23~2585)

**k-NearestNeighbors(KNN)**

LeKNNestunalgorithmedeclassificationnonparamétrique.Ilclassifielesnouveaux
échantillonsenfonctiondeskplusprochesvoisinsdansl'espacedescaractéristiques.

```
from sklearn.neighbors import KNeighborsClassifierknn =
KNeighborsClassifier(n_neighbors= 5 ) knn.fit(X_tfidf, y_train) y_pred_knn
= knn.predict(X_tfidf_test) accuracy_knn = accuracy_score(y_test,
y_pred_knn)
```
**RandomForest**

LeRandomForestestunensembledeméthodesquiconstruitplusieursarbresdedécision
etcombineleursprédictionspouraméliorerlaperformanceetréduirelesurapprentissage.

```
from sklearn.ensemble import RandomForestClassifier rf =
RandomForestClassifier(n_estimators= 100 ) rf.fit(X_tfidf, y_train)
y_pred_rf = rf.predict(X_tfidf_test) accuracy_rf =
accuracy_score(y_test, y_pred_rf)
```
```
UtilisationdeGridSearchCVavecRandomForest
```
```
Afind'optimiserlesperformancesdenotremodèleRandomForest,nousavonsutilisé
GridSearchCVpourrechercherlesmeilleurshyperparamètres.GridSearchCVeffectueune
rechercheexhaustivesurunespécificationdegrilledeparamètres,enutilisantlavalidation
croiséepourévaluerchaquecombinaisondeparamètres.
```
```
MeilleursParamètresetPerformanceduModèle
```
```
Aprèsl'optimisation,lesmeilleursparamètresobtenussontlessuivants:
● `n_estimators`: 200
● `max_depth`:None
● `min_samples_split`: 5
● `min_samples_leaf`: 1
```
```
Grâceàcesparamètresoptimisés,laprécisiondumodèleRandomForestaétéaméliorée
à 8 pointsenviron(de0.71à0.79).
CetteméthodeaaussiétéutiliséesurKNN
```
**VotingClassifier**


LeVotingClassifiercombinelesprédictionsdeplusieursmodèlesdebasepouraméliorer
lesperformancesglobales.Nousavonsutiliséunvotedetype"hard"oùchaquemodèle
contribuedemanièreégaleàlaprédictionfinale.

```
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('logreg', logreg), ('knn',
knn), ('rf', rf) ], voting='hard') voting_clf.fit(X_tfidf, y_train)
y_pred_voting = voting_clf.predict(X_tfidf_test) accuracy_voting =
accuracy_score(y_test, y_pred_voting)
```
6. **Comparatif**

```
Lesperformancesdesdifférentsmodèlesontétécomparéesentermesdeprécision
(accuracy)surl'ensembledetest.Lesrésultatssontprésentésdansletableauci-dessous:
```
**Conclusion**

Àpartirdesrésultatsobtenus,nousobservonsqueleVotingClassifieroffrelesmeilleures
performancesavecuneprécisionde0.80.Cemodèlecombinelesforcesdeplusieurs
algorithmesdebase,cequiluipermetdemieuxgénéralisersurlesdonnéesnonvues.


**Bibliographie**

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

[http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/notebooks/classification_multiple.html](http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/notebooks/classification_multiple.html)

https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-u
sing-pickle

https://monkeylearn.com/blog/what-is-tf-idf




