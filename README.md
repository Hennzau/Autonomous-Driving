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

Voici les différents éléments de ce repository : 

- oasis_agent.yaml : Il s'agit du graph du projet. DORA analyse ce graph et éxécute tout comme il faut. DORA peut aussi afficher les différents opérateurs présents dans le projet et les liens qui existent entre les opérateurs

- operators/oasis_agent : il s'agit de l'opérateur le plus important. En réalité il ne s'agit pas d'un opérateur comme les autres mais d'un custom-node. C'est à dire que contrairement aux opérateurs, il ne fait pas que recevoir des informations et d'en renvoyer, c'est un programme éxécutable à part entière qui a le droit de communiquer avec le réseau d'opérateurs. Ici le module oasis_agent.py est éxécuté par DORA, il se charge de communiquer avec le serveur CARLA, de charger le scénario, de charger notre agent my_agent.py et de controler le véhicule du scénario avec le code de my_agent.

- operators/oasis_agent/my_agent : afin de controler le véhicule du scénario, my_agent doit renvoyer un objet du type VehicleControl, qui modélise l'accélérateur du véhicule, les freins du véhicule etc... Il est également chargé de recevoir les informations des capteurs établis sur la voiture et de les communiquer dans le réseau afin qu'ils soient récupéré par les bons opérateurs et traités correctement.

- operators/yolov_filter : Directement "branché" après l'opérateur principal, il s'occupe de traiter la première image issue de la caméra afin d'en extraire les informations pertinentes à l'aide du machine learning. Il renvoie des informations importantes sur l'existence et l'emplacement sur la caméra des éléments extérieurs.

- operators/plot : Cet opérateur récupère divers informations (actuellement l'image et les informations de yolov_filter) et les affiche sur une fenêtre 640x640 (je ne comprend pas ce qui coince quand je veux mettre plus grand...)

## Lancement

Pour lancer le programme, après avoir procédé à un Dora Up et avoir allumé le serveur, il faut lancer Dora Start. Vous devriez voir quelque chose dans le terminal, il s'agit d'un identifiant de notre lancement. ça permet de débugger et d'accéder aux logs.

![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/6.png)

 Le premier lancement peut mettre très longtemps à s'éxécuter, en fait c'est le serveur Carla qui coince : la map met très longtemps à charger la première fois (elle est ensuite mise en cache). Après une trentaine de seconde la première fois vous devriez observer une fenêtre qui s'ouvre, mais pas en premier plan, vous pouvez y accéder depuis la barre des tâches en bas :

 ![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/7.png)
 ![](https://github.com/Hennzau/Autonomous-Driving/blob/main/docs/8.png)
 
