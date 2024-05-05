# Descripció

El desafiament GLUP24 es basa en la predicció dels nivells de glucosa en sang de persones amb diabetis tipus 1. Aquesta condició mèdica implica un desequilibri en els nivells de glucosa en el cos que requereix un monitoratge constant i un tractament adequat per evitar complicacions. L’objectiu principal d’aquest treball és desenvolupar un algoritme que sigui capaç de predir els nivells de glucosa en el cos humà durant intervals específics de temps. Aquesta predicció és fonamental per millorar la qualitat de vida de les persones amb diabetis i permet un control més efectiu de la malaltia i una optimització del tractament.

Per resoldre el problema, es disposa d’un conjunt de dades recopilades de sis individus amb diabetis tipus 1 que han utilitzat teràpia de bomba d’insulina amb monitors continus de glucosa i polseres d’activitat física. Aquestes dades inclouen mesures de glucosa en sang, dosis d’insulina, hores de menjar, exercici, son, feina, estres i malaltia. També inclouen dades d’activitat fisiològica com la freqüència cardíaca, la resposta galvànica de la pell o la temperatura, la temperatura de l’aire i el recompte de passos.

# Instal·lació

Per executar aquest codi, assegureu-vos de tenir Python 3.9.13 instal·lat al vostre sistema. A continuació, podeu instal·lar les dependències necessàries utilitzant el fitxer `requirements.txt`. Executeu la següent comanda en la terminal per instal·lar les dependències:

```
pip install -r requirements.txt
```

Abans d'executar cap fitxer, s'ha d'inicialitzar un servidor de MLflow. Si utilitzeu l'IDE de VSCode, podeu fer-ho executant la tasca "Start MLFLow". Aquesta tasca utilitza els fitxers de la carpeta MLFlow. També podeu iniciar el servidor amb la comanda següent:

```
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./MLflow/mlruns --default-artifact-root ./MLflow/mlartifacts
```

El host i el port han de ser els mateixos especificats a l'arxiu `.env`.

# Estructura

La distribució de les carpetes del codi és la següent:

- **.gitattributes**
- **.gitignore**
- **evaluate.py**: Script per avaluar el rendiment dels models sense guardar els resultats.
- **example.env**: Exemple de fitxer d'entorn.
- **main.py**: Fitxer principal per executar el codi principal.
- **optimize.py**: Script per executar l'optimització dels híper-paràmetres.
- **requirements.txt**: Fitxer que conté les dependències del projecte.
- **test.py**: Script per avaluar el model guardant els resultats a MLflow.
- **train.py**: Script per entrenar els models.
- **training_parameters.json**: Fitxer JSON que conté els paràmetres d'entrenament utilitzats pel fitxer train.py.

- **.vscode**: Carpeta que conté configuracions per a l'entorn de desenvolupament de Visual Studio Code.

  - **launch.json**: Configuració per a la posada en marxa del depurador.
  - **tasks.json**: Configuració de tasques específiques per a Visual Studio Code.

- **Datasets**: Carpeta que conté els conjunts de dades utilitzats en el projecte.

  - **columns_description.txt**: Descripció de les columnes del conjunt de dades.
  - **Dataset.zip**: Fitxer ZIP que conté els conjunts de dades.
  - Carpeta per a cada conjunt de dades, numerades per identificar-los.
    - **{num}**: Carpeta del conjunt de dades amb el número identificador.
      - **{num}\_test.csv**: Conjunt de dades de prova.
      - **{num}\_train.csv**: Conjunt de dades d'entrenament.

- **Optimization**: Carpeta que conté els resultats de la cerca d'hiperparàmetres.

  - Fitxers de base de dades que emmagatzemen els resultats de la cerca d'hiperparàmetres.

- **Misc**: Carpeta que conté scripts i utilitats addicionals.

  - **callbacks.py**: Script que conté els callbacks personalitzats.
  - **columns.py**: Script que conté les descripcions de les columnes.
  - **plot.py**: Script per a la visualització de dades.
  - **utils.py**: Script que conté funcions d'utilitat.

- **Models**: Carpeta que conté els scripts dels models utilitzats en el projecte.
  - **model.py**: Script per seleccionar un model.
  - **CNN.py**: Implementació del model CNN (Xarxa Neuronal Convolucional).
  - **GRU.py**: Implementació del model GRU (Unitat Recurrent Gated).
  - **LSTM.py**: Implementació del model LSTM (Xarxa de Memòria de Llarga Recurrència).
  - **RNN.py**: Implementació del model RNN (Xarxa Neuronal Recurrent).

# Funcionament

Els arxius principals són `train.py`, `optimize.py`, `test.py` i `evaluate.py`.

`train.py`: És responsable d'entrenar un model d'aprenentatge automàtic en un conjunt de dades específic, avaluar-ne el rendiment i registrar els resultats juntament amb el model a MLflow.

`optimize.py`: És un script que utilitza Optuna per optimitzar els hiperparàmetres d'un model d'aprenentatge automàtic. Carrega les variables d'entorn des d'un arxiu .env, defineix una funció objectiu per a l'optimització dels hiperparàmetres i executa l'optimització utilitzant Optuna.

`test.py`: Avalua el model entrenat utilitzant dades de prova. Primer, es connecta al servidor de MLflow i obté el model entrenat corresponent a l'ID de l'execució. A continuació, carrega les dades de prova, les pre-processa i les utilitza per avaluar el model. Finalment, calcula diverses mètriques d'avaluació, com ara l'error mitjà absolut, l'error mitjà quadràtic, etc., i les registra a MLflow juntament amb gràfics de comparació entre les prediccions i els valors reals.

`evaluate.py`: Avalua el model entrenat en un conjunt de dades de prova, però no registra les mètriques a MLflow. Primer, connecta amb el servidor de MLflow i obté el model entrenat corresponent a l'ID de l'execució. Llavors, carrega les dades de prova, les pre-processa i les utilitza per avaluar el model. Finalment, genera un gràfic de comparació entre les prediccions i els valors reals de les dades de prova, però no registra les mètriques obtingudes.

La seqüència de passos és la següent:

1. Executar l'arxiu `train.py` o `optimize.py` per obtenir un model entrenat amb els paràmetres especificats a l'arxiu `training_parameters.json` i el tipus de model i conjunt de dades de l'entorn (.env).
2. Un cop entrenat el model, si s'han obtingut bons resultats, extreure l'ID de l'entrenament de MLflow.
3. A partir d'aquest ID, executar la predicció de les dades amb l'arxiu `test.py`.

El directori Models conté els scripts dels models utilitzats en el projecte, mentre que el directori Misc conté utilitats addicionals com scripts de visualització de dades i callbacks personalitzats. El directori Optimization emmagatzema els resultats de la cerca d'hiperparàmetres, mentre que Datasets conté els conjunts de dades utilitzats, amb una carpeta per a cada conjunt numerada per identificar-los.

En resum, aquest projecte ofereix una eina completa per a la predicció dels nivells de glucosa en sang en pacients amb diabetis tipus 1, amb la capacitat de entrenar, avaluar i optimitzar models d'aprenentatge automàtic, tot gestionat i rastrejat amb MLflow.
