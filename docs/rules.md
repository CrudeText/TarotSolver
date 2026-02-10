# Règlement officiel du Tarot (FFT) — Résumé pour implémentation

Source : **Règlement officiel du jeu de Tarot**, Fédération Française de Tarot (version 1er juillet 2012).  
Référence locale : `Règlement Tarot.pdf`.

Ce document est un résumé structuré pour l’implémentation du moteur de jeu. Chaque section peut être cochée au fur et à mesure.

---

## 1. Le jeu (commun à 3, 4 et 5 joueurs)

### 1.1 Les cartes

- [ ] **78 cartes** au total.
- [ ] **4 couleurs** : Pique, Cœur, Carreau, Trèfle. Chaque couleur a **14 cartes** :
  - Honneurs : Roi, Dame, Cavalier, Valet (ordre décroissant de force).
  - Puis : 10, 9, 8, 7, 6, 5, 4, 3, 2, As.
- [ ] **21 Atouts** numérotés : du plus fort (21) au plus faible (1). Le 1 est appelé **Petit**.
- [ ] **L’Excuse** : carte à part (étoile / mandoline). Elle dispense de fournir la couleur ou l’atout demandé ; ne remporte pas la levée (sauf en cas de Chelem). Ne doit pas être jouée à la dernière levée (sinon elle change de camp), sauf en cas de Chelem.
- [ ] **Les 3 Bouts (Oudlers)** : **Excuse**, **1 (Petit)** et **21**. Les Bouts déterminent le nombre de points à réaliser pour gagner le contrat. Le **1** et le **21** sont des atouts (le 1 est le plus faible, le 21 le plus fort) ; l’**Excuse** est une carte à part (pas un atout). Le **Petit (1)** a en plus la règle du **Petit au Bout** (prime si il est dans la dernière levée).

### 1.2 Valeur des cartes (pour le décompte)

On compte les cartes **deux par deux** :

| Association | Points |
|-------------|--------|
| 1 Oudler (21, Petit ou Excuse) + 1 petite carte | 5 |
| 1 Roi + 1 petite carte | 5 |
| 1 Dame + 1 petite carte | 4 |
| 1 Cavalier + 1 petite carte | 3 |
| 1 Valet + 1 petite carte | 2 |
| 2 petites cartes (couleur ou atout) | 1 |

**Total : 91 points** par donne.

Valeur unitaire (pour moitiés de points en 3p/5p) : Oudler 4,5 ; Roi 4,5 ; Dame 3,5 ; Cavalier 2,5 ; Valet 1,5 ; autre 0,5.

### 1.3 Points à réaliser (contrat du preneur)

Le preneur doit atteindre un **minimum de points** dans ses levées, selon le **nombre de Bouts** qu’il détient dans ses plis en fin de donne :

| Bouts du preneur | Points minimum à réaliser |
|------------------|---------------------------|
| Sans Oudler | 56 |
| Avec 1 Oudler | 51 |
| Avec 2 Oudlers | 41 |
| Avec 3 Oudlers | 36 |

- [ ] Implémenter ce tableau et l’utiliser pour savoir si le contrat est réussi ou chuté.
- [ ] En **Garde sans le Chien** : un éventuel Bout au Chien compte pour le preneur.

---

## 2. Jeu à 4 joueurs (priorité d’implémentation)

### 2.1 Principe

- [ ] Un joueur (le **preneur**) joue seul contre les **trois défenseurs** (équipe Défense).
- [ ] L’association ne dure qu’une donne.

### 2.2 Distribution

- [ ] Déterminer le **donneur** : tirage au sort (plus petite carte ; en cas d’égalité, priorité Pique > Cœur > Carreau > Trèfle ; l’As de Trèfle est la plus petite ; les Atouts sont prioritaires, l’Excuse ne compte pas).
- [ ] Le joueur en face du donneur bat le jeu. Le voisin de gauche **coupe** (prend ou laisse obligatoirement plus de 3 cartes) et referme.
- [ ] Distribution **3 par 3**, sens **contraire des aiguilles d’une montre**.
- [ ] Pendant la distribution, le donneur constitue un **Chien de 6 cartes**. Interdit de mettre la **première** ou la **dernière** carte du paquet au Chien.
- [ ] Chaque joueur reçoit **18 cartes**.
- [ ] À la fin, le donneur montre qu’il a bien mis 6 cartes au Chien. Personne ne touche ses cartes avant la fin de la distribution.
- [ ] **Petit sec** : si un joueur n’a que le Petit comme atout (et pas l’Excuse), il doit l’annoncer, étaler son jeu et la donne est annulée ; le voisin de droite du donneur redistribue.
- [ ] Donneur suivant : à tour de rôle dans le sens du jeu.

### 2.3 Les enchères

- [ ] Le joueur **à droite du donneur** parle en premier. Puis tour à tour vers la droite jusqu’au donneur.
- [ ] Chaque joueur ne parle **qu’une seule fois**. Options : **Passer** ou enchère (Prise, Garde, Garde sans le Chien, Garde contre le Chien). Un joueur à droite peut **couvrir** par une enchère supérieure.
- [ ] Ordre croissant des enchères : **Prise (Petite)** < **Garde** < **Garde sans le Chien** < **Garde contre le Chien**.
- [ ] Si les quatre passent : nouvelle distribution par le voisin de droite du donneur.
- [ ] Sur contrat Garde Sans ou Garde Contre, le joueur doit appeler l’arbitre (en logiciel : on enregistre le contrat).

### 2.4 Le Chien et l’Écart (Prise ou Garde)

- [ ] Le donneur tend le Chien au preneur. Le preneur **retourne les 6 cartes** du Chien pour que tout le monde les voie.
- [ ] Il les **incorpore à son jeu**, puis **écarte 6 cartes** (l’Écart). L’Écart reste secret jusqu’à la fin ; il sera compté avec les levées du preneur.
- [ ] Règles d’écart : on ne peut écarter **ni Roi ni Bout**. On n’écarte des Atouts que si indispensable, et en les montrant à la Défense.
- [ ] Le preneur montre qu’il a bien mis 6 cartes à l’Écart, puis dit **« Jeu »**. Une fois « Jeu » dit et le nombre de cartes conforme (ou dès la première carte jouée), l’Écart ne peut plus être modifié ni consulté.
- [ ] Le preneur peut rectifier son Écart tant qu’aucune carte n’est jouée (même s’il a dit « Jeu »).

### 2.5 Garde sans le Chien / Garde contre le Chien

- [ ] **Garde sans** : le Chien reste **face cachée**. Il est placé devant le preneur et sera compté avec ses levées. Le preneur ne fait **pas** d’écart (il ne prend pas le Chien dans sa main).
- [ ] **Garde contre** : le Chien reste **face cachée**. Il est donné au défenseur **en face du preneur** et sera compté avec les levées de la Défense.

### 2.6 Poignée (4 joueurs)

- [ ] **Simple** : 10 Atouts → prime 20 points. **Double** : 13 Atouts → 30 points. **Triple** : 15 Atouts → 40 points.
- [ ] L’Excuse peut remplacer un Atout manquant ; si l’Excuse est dans la Poignée, l’annonceur n’a pas d’autre Atout.
- [ ] Si le joueur a 11, 12, 14 ou 16 Atouts, il doit en cacher un ou plus. Si le preneur a 4 Rois et 15 atouts, l’atout écarté doit être remontré avec la triple Poignée.
- [ ] Annonce et présentation **juste avant de jouer sa première carte**, atouts classés décroissant, en une seule fois.
- [ ] La prime est acquise au **camp vainqueur** de la donne (donnée par le camp perdant à chaque adversaire).

### 2.7 Petit au Bout

- [ ] Si le **Petit** est dans la **dernière levée**, le camp qui fait cette levée a le Petit au Bout.
- [ ] Prime : **10 points**, multipliée par le coefficient du contrat. Comptée quel que soit le résultat de la donne (gagnée ou chutée). En cas de gain du preneur, la prime est déduite de ses gains si c’est la Défense qui a le Petit au Bout (et inversement).

### 2.8 Chelem

- [ ] **Chelem** = faire toutes les levées.
- [ ] Annoncé après l’annonce du contrat. En cas d’annonce, l’entame revient au joueur qui a demandé le Chelem.
- [ ] Primes / pénalités : Chelem annoncé et réalisé +400 ; non annoncé mais réalisé +200 ; annoncé mais non réalisé −200. Si la Défense inflige un Chelem au preneur : +200 à chaque défenseur.
- [ ] En cas de Chelem réussi, le demandeur doit faire tous ses plis ; s’il a l’Excuse, la jouer en dernier (le Petit compte alors au Bout s’il est mené à l’avant-dernier pli).

### 2.9 Jeu de la carte

- [ ] **Entame** : joueur à **droite du donneur**. Puis sens **inverse des aiguilles d’une montre**. Le joueur qui remporte une levée entame la suivante.
- [ ] **À l’atout** : obligation de **monter** sur le plus fort atout déjà en jeu (même partenaire). Si on ne peut pas monter, on joue un atout de son choix (en général le plus petit) : « pisser ».
- [ ] **À la couleur** : obligation de **fournir** la couleur demandée, pas de monter.
- [ ] **Couper** : si pas de carte de la couleur, obligation de jouer atout. Si le précédent a coupé, obligation de **surcouper** ou de **sous-couper** (pisser) si on ne peut pas surcouper.
- [ ] **Défausse** : si ni la couleur ni atout, on joue une carte de son choix.
- [ ] Si la première carte d’une levée est l’**Excuse**, c’est la **carte suivante** qui détermine la couleur.
- [ ] **Excuse** : ne remporte pas la levée (sauf Chelem). Reste au camp qui la détient : celui qui l’a jouée récupère l’Excuse dans son tas de plis et donne en échange une carte sans valeur à l’autre camp. Si son camp n’a encore aucun pli, il met l’Excuse en évidence et donnera une carte dès qu’il en aura l’occasion.
- [ ] Tant qu’une levée n’est pas ramassée, on peut consulter la levée précédente. Les plis sont ramassés par le camp qui remporte la levée (en face du preneur pour la Défense).

### 2.10 Calcul des scores (4 joueurs)

- [ ] À la fin, on compte les points dans les levées du preneur d’une part, de la Défense d’autre part.
- [ ] **Réussite** : points ≥ minimum (selon nombre de Bouts). **Juste fait** = exactement le minimum (contrat « gagné de 0 »). **Chute** : points &lt; minimum.
- [ ] **Formule** :  
  - (points de gain ou de perte) + **25** (valeur arbitraire du contrat)  
  - Puis multiplier par le **coefficient du contrat** :  
    - Prise : ×1  
    - Garde : ×2  
    - Garde sans le Chien : ×4  
    - Garde contre le Chien : ×6  
  - Puis ajouter les primes (Poignée, Petit au Bout, Chelem).

Donc : **base = (différence par rapport au contrat + 25) × coefficient**, puis + primes.

- [ ] **Marque en donnes libres** : chaque défenseur marque le même nombre (négatif si le preneur gagne, positif s’il chute). Le preneur marque **3 fois** ce total (positif s’il gagne, négatif s’il chute). **Somme des 4 scores = 0.**

**Pas d’égalité possible** : en cas d’exactement le minimum, le contrat est « juste fait » (gagné de 0) → on ajoute 25, on multiplie, etc. Les scores sont toujours différenciés par les primes et la répartition preneur/défense.

---

## 3. Jeu à 3 joueurs

- [ ] Règle quasi identique à 4. Contrats identiques.
- [ ] Distribution **4 par 4**, Chien **6 cartes**. Chaque joueur reçoit **24 cartes**.
- [ ] Poignées : **Simple 13 atouts**, **Double 15**, **Triple 18**. Primes inchangées (20, 30, 40).
- [ ] Décompte au **½ point** près. En cas de non-entier, le ½ point va au camp gagnant (après que le camp ayant l’Excuse ait donné une carte). Ex. : 40,5 avec 2 Bouts → 40 pour le preneur → chute de 1 ; 41,5 avec 2 Bouts → 42 pour le preneur → gain de 1.
- [ ] Marque : défenseurs comme en 4p ; **preneur × 2**. Total des 3 scores = 0.

---

## 4. Jeu à 5 joueurs

- [ ] Un joueur est **le mort** : il distribue aux 4 autres et ne joue pas. Il ne compte pas pour les scores de la donne. Chacun est mort à tour de rôle (sens du jeu) tant que personne ne prend.
- [ ] Distribution **3 par 3**, Chien **3 cartes**. Chaque joueur reçoit **15 cartes**.
- [ ] Contrats identiques. Après les enchères, **avant de retourner le Chien**, le preneur **appelle un Roi** (ou Dame si 4 Rois, ou Cavalier si 4 Rois et 4 Dames, ou Valet si 4 grands mariages ; ou s’appeler lui-même avec un Roi de son jeu en jeu exceptionnel).
- [ ] Si la carte appelée est au Chien ou si le preneur s’est appelé : **preneur seul contre 4**. Sinon le détenteur de la carte appelée est son **partenaire** : **2 contre 3**.
- [ ] Entame : interdite dans la couleur de la carte appelée, sauf si c’est la carte appelée elle-même. Le partenaire ne révèle pas son statut avant d’avoir joué le Roi appelé.
- [ ] Poignées : **Simple 8 atouts**, **Double 10**, **Triple 13**. Primes inchangées.
- [ ] Décompte au ½ point comme en 3p.
- [ ] Marque : défense comme en 4p. Attaquants : **2/3 au preneur, 1/3 au partenaire**. Si preneur seul (1 contre 4), il prend toute la marque. Total des 5 scores = 0.

---

## 5. Match et classement (pour notre usage)

- [ ] **Match** = N donnes (paramètre du tournoi). On additionne les **scores de chaque joueur** sur les N donnes (scores FFT : preneur ×3 ou ×2 en 3p, défenseurs négatifs/positifs, etc.).
- [ ] **Classement du match** : par **total de points décroissant**. Pas d’égalité réglementaire au tarot (juste fait = +25 dans la formule) ; en cas d’égalité de totaux sur le match, définir un critère (ex. nombre de donnes gagnées, ou partage).
- [ ] Pour les tournois « donnes libres » : classement par addition des points (ou des bilans par position) sur toutes les donnes.

---

## 6. Checklist d’implémentation (ordre suggéré)

1. [ ] Deck : 78 cartes, couleurs, atouts, Excuse, valeur des cartes.
2. [ ] Distribution 4p : 18 cartes/joueur, Chien 6, règles du donneur et du Petit sec.
3. [ ] Enchères 4p : ordre, Prise/Garde/Garde sans/Garde contre, une parole par joueur.
4. [ ] Chien et Écart : prise du Chien, écart 6 cartes, règles (pas de Roi ni Bout, atouts si nécessaire).
5. [ ] Garde sans / Garde contre : Chien non pris par le preneur, attribution des points du Chien.
6. [ ] Jeu de la carte : entame, ordre de jeu, règles atout/couleur/coupe/défausse, Excuse, levées.
7. [ ] Décompte : 91 points, Bouts, points à réaliser, juste fait / gain / chute.
8. [ ] Calcul des scores : (écart + 25) × coefficient + Poignée + Petit au Bout + Chelem.
9. [ ] Marque 4p : preneur ×3, chaque défenseur même score, somme = 0.
10. [ ] Poignée et Petit au Bout (annonces, primes).
11. [ ] Chelem (annonce, entame, primes).
12. [ ] 3 joueurs : distribution, poignées, ½ point, marque ×2 preneur.
13. [ ] 5 joueurs : mort, distribution, appel du Roi, 2v3, poignées, ½ point, répartition 2/3–1/3.

---

*Document généré pour le projet Tarot Solver. Référence : Règlement Tarot.pdf (FFT, juillet 2012).*
