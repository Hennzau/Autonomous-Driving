# Autonomous-Driving
2023 IEEE &amp; OpenAtom Competition

This repo is the work of LE VAN Enzo, VALLADE Antoine and LUCAS Chloé, students of CentraleSupélec Engineering School in France.

# Programmation avec l'environnement proposé

## Démarrage des différents éléments
Cette section a pour but d'expliquer comment développer une voiture autonome avec le simulateur CARLA dans l'environnement proposé par ce repository.

Tout d'abord, le développement doit se faire sur PyCharm Community Edition, c'est un outil très simple, gratuit. Lors de son ouverture, vous êtes normalement déjà dans le projet Autonomous-Driving. Pour ouvrir un terminal si ce n'est pas le cas, cliquez en bas à gauche de votre écran : 
![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/1.png)

Il s'agit ensuite de démarrer les différents logiciels : CARLA et DORA. Carla c'est le simulateur, DORA c'est le logiciel qui va éxécuter tout notre code. Pour DORA, il faut en réalité démarrer le processus avant de lui donner notre code à éxécuter. Cela se fait depuis les scripts déjà enregistrés dans PyCharm en haut à droite (lancer Carla Server et Dora Up):

![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/2.png)
![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/3.png)

Une fois Dora Up lancé, vous devriez voir un message vous indiquant les deux processus qui ont été lancés : 

![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/4.png)

Après quelques instants, CARLA Server devrait être opérationnel, vous aurez donc une fenêtre avec une ville. Vous pouvez diminuer cette fenêtre on n'en a pas besoin par la suite 

![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/5.png)

## Développement 

Le développement de cette voiture sur simulateur se fait sous deux axes : du code Python, et la description d'un graph. En fait DORA est un framework qui permet de créer de manière isolé des opérateurs (operator) qui reçoivent des entrées (inputs) et renvoient des sorties (outputs). La description du graphe nous permet alors de rediriger les sorties et les entrées sur les différents opérateurs que l'on créer.

Par exemple il y'a un opérateur "yolov_filter" qui s'occupe de récupérer une image, et qui nous renvoie cette image accompagnée d'une liste contenant les différents objets que yolov a réussi à identifier (cela consiste en "dans cette zone là de l'image je vois une voiture, dans cette zone là je vois un panneau stop"...). Et bien notre jeu dans le graph va être de lui donner la bonne image que l'on souhaite, et de renvoyer ses sorties à d'autres opérateurs, ou comme actuellement, a un opérateur de rendu, qui nous permet d'afficher les différentes données de notre voiture afin de monitorer son développement.


 
