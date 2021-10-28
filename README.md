# Face Detection e Face Recognition utilizzando le CNN

## Face Detection

Face Detection utilizzando VGGFace2:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13gdTneEgbzn0SvvzntR7t2RDhKsvYPZW?usp=sharing)

### VGGFace
Il modello VGGFace è stato descritto da Omkar Parkhi nella tesi del 2015 intitolata [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).  
Una parte del documento era una descrizione di come sviluppare un dataset di addestramento molto ampio, necessario per addestrare i moderni sistemi di riconoscimento facciale basati su CNN, per competere con i grandi dataset utilizzati per addestrare i modelli di Facebook e Google.  
Per quanto riguarda la CNN di VGGFace in sé, l'architettura adottata la rende molto profonda, con ben 37 layer, divisi in blocchi di layer convoluzionali con kernel piccoli e attivazioni ReLU, seguiti da layer di max pooling con, alla fine del classificatore, layer full connected:  

<img src='https://drive.google.com/uc?id=1ppG-bZ_032RiUqWH9IOSN3pRtO1iDO4g'>

### VGGFace2
In una tesi del 2017 intitolata “[VGGFace2: A dataset for recognising faces across pose and age](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)”, Qiong Cao, del VGG, descrive un lavoro di miglioramento del precedente VGGFace.  
Descrivono VGGFace2 come un set di dati molto più ampio che hanno raccolto allo scopo di addestrare e valutare modelli di riconoscimento facciale ancora più efficaci.
Il dataset contiene 3,31 milioni di immagini di 9131 soggetti, con una media di 362,6 immagini per ogni soggetto.  
Due dei modelli addestrati su questo dataset, sono ResNet-50 e SqueezeNet-ResNet-50 (chiamato SE-ResNet-50 o SENet), e sono le variazioni di [questi modelli](https://github.com/ox-vgg/vgg_face2) che sono state rese disponibili dagli autori , insieme al codice associato.  

## Face Recognition

Face Recognition utilizzando MTCNN:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hd7IoFsZBFw1uB98gu4IsFsUb5LBphsP?usp=sharing)

### MTCNN
Il progetto [ipazc/MTCNN](https://github.com/ipazc/mtcnn) fornisce un'implementazione dell'architettura MTCNN utilizzando TensorFlow e OpenCV. I due principali benefici di questo progetto sono:
    
*   la possibilità di utilizzare un modello performante e pre-addestrato;
*   la possibilità di installare il modello come una libreria pronta all'uso nel proprio codice.

Tecnicamente MTCNN è composto da un prepassaggio e 3 CNN, non direttamente collegate tra di loro. Per prima cosa l'immagine viene ridimensionata a scale diverse per costruire una image pyramid, che sarà l'input della prima delle reti, dopodichè le reti in ordine:
    
### 1. Proposal Network (P-Net)
É una fully connected network (FCN). Meno profonda delle prossime CNN, viene utilizzata per ottenere finestre candidate a contenere volti.  

<img src='https://drive.google.com/uc?id=1XQKluW0_5pru_-X80OJduz9_7kXbzjVg'>
    
### 2. Refine Network (R-Net)
Tutte le finestre vengono inviate a questa CNN che riduce ulteriormente il numero di candidati a contenere un viso, esegue una bounding box regression e utilizza la non-maxiam suppression (NMS) per unire i candidati sovrapposti. L'ouput sarà la conferma della presenza di effetivo volto.  

<img src='https://drive.google.com/uc?id=1ymIh--XX_fdvmclC10xLJIGAznb6xq4A'>
    
## 3. Output network (O-Net)
Questo passaggio è simile alla R-Net, ma il suo obiettivo è di descrivere il viso più nel dettaglio individuando i 5 facial landmarks.  

<img src='https://drive.google.com/uc?id=1nrOnIpqFCif5sI8VFZE8dm7wH2kghvh9'>
