import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from math import floor
import os

def animer(courbe, listeLValeurs, listeLCouleurs, garder):
    def update(frame):
        nonlocal listeLXY, listeLC

        t = (nbX-1) * (frame / nbFrames)
        i = int(t)
        dx = (t - i)
        if i+1 < nbX:
            lXY = listeLValeurs[i] + dx * (listeLValeurs[i+1] - listeLValeurs[i])
            lC = listeLCouleurs[i] + dx * (listeLCouleurs[i+1] - listeLCouleurs[i])
        else:
            lXY = listeLValeurs[i]
            lC = listeLCouleurs[i]

        lPoints = np.vstack([listeLXY, lXY])
        lCouleurs = np.concatenate([listeLC, lC])
        courbe.set_offsets(lPoints)
        courbe.set_array(lCouleurs)
        if dx < 1e-5 and garder:
            listeLXY = lPoints
            listeLC = lCouleurs
        return (courbe,)

    listeLXY, listeLC = np.empty((0, 2)), np.empty(0)
    nbX = nbGenerations
    nbFramesParGen = 10
    nbFrames = nbFramesParGen*(nbX-1)
    return FuncAnimation(fig, update, frames=nbFrames+1, interval=int(1000/nbFramesParGen), blit=(mode==1), repeat=False)

def afficher():
    global animation1, animation2, animation3

    listeLValeurs = listeLValeursNormales if normal else listeLValeursJoueurs
    if mode == 0:
        axe1.clear()
        axe2.clear()
        lX = listeLValeurs[:, :, 0].flatten()
        lY1 = listeLValeurs[:, :, 2].flatten()
        lY2 = listeLValeurs[:, :, 3].flatten()
        courbe1 = axe1.scatter(lX, lY1, c=lY1, cmap=cmapPersonnel)
        courbe2 = axe2.scatter(lX, lY2, c=lY1, cmap=cmapPersonnel)
        if animations :
            if animation1 is not None and animation1.event_source is not None:
                animation1.event_source.stop()
                animation1 = None
            animation1 = animer(courbe1, listeLValeurs[:, :, [0,2]], listeLValeurs[:, :, 2], garder=True)
            if animation2 is not None and animation2.event_source is not None:
                animation2.event_source.stop()
                animation2 = None
            animation2 = animer(courbe2, listeLValeurs[:, :, [0,3]], listeLValeurs[:, :, 2], garder=True)
        axe1.set_xlim(-0.5, nbGenerations-0.5)
        axe1.set_ylim(0)
        axe1.set_xlabel("Generations")
        axe1.set_ylabel("Score")
        axe2.set_xlim(-0.5, nbGenerations-0.5)
        axe2.set_ylim(0)
        axe2.set_xlabel("Generations")
        axe2.set_ylabel("nbCoups")
    elif mode == 1:
        axe3.clear()
        iV1, iV2 = sliderV1.val-1, sliderV2.val-1
        listeLPoids = listeLPoidsNormals if normal else listeLPoidsJoueurs
        lX = listeLPoids[:, :, iV1].flatten()
        lY = listeLPoids[:, :, iV2].flatten()
        lC = listeLValeurs[:, :, 2].flatten()
        courbe3 = axe3.scatter(lX, lY, c=lC, cmap=cmapPersonnel)
        if animations :
            if animation3 is not None and animation3.event_source is not None:
                animation3.event_source.stop()
                animation3 = None
            animation3 = animer(courbe3, listeLPoids[:, :, [iV1, iV2]], listeLValeurs[:, :, 2], garder=False)
        axe3.set_xlim(listeLPoids[:, :, iV1].flatten().min(), listeLPoids[:, :, iV1].flatten().max())
        axe3.set_ylim(listeLPoids[:, :, iV2].flatten().min(), listeLPoids[:, :, iV2].flatten().max())
        axe3.set_xlabel("Input 1")
        axe3.set_ylabel("Input 2")
    
    fig.canvas.draw_idle()

def changer_mode_normal(_, visuel=True):
    global normal
    normal = not normal
    boutonNormal.label.set_text("Joueurs ? " if normal else "Normal ?")
    if visuel:
        afficher()

def changer_mode_animations(_, visuel=True):
    global animations
    animations = not animations
    boutonAnimations.label.set_text("Sans animations ? " if animations else "Avec animations ?")
    if visuel :
        afficher()

def changer_mode(_, visuel=True):
    global mode
    mode = 1-mode
    boutonMode.label.set_text("Mode 1 ?" if mode==0 else "Mode 0 ?")
    for lPos, axe in zip(lPositionsAxes, lAxes):
        if lPos[mode] is not None:
            axe.set_position(lPos[mode])
            axe.set_visible(True)
        else:
            axe.set_visible(False)
    if visuel :
        afficher()

def changer_valeur_slider(_, slider, visuel=True):
    val = slider.val
    if visuel:
        afficher()

nomDossier = os.path.dirname(__file__)
nomFichier = os.path.join(nomDossier, "GA_2_v2.txt")
with open(nomFichier, 'r') as fichier:
    listeLRes = eval(fichier.read())
(nbJoueurs, nbBalles, nbGenerations, nbParties, lCouches, kNN), listeLRes, meilleurNN = listeLRes
nbGenerations = len(listeLRes)
listeLValeursNormales = np.array([[(iG, vraiIJ, score, nbCoups) for vraiIJ, score, nbCoups, dicoNN in lRes] for iG,lRes in enumerate(listeLRes)])
listeLPoidsNormals = np.array([[dicoNN['0.weight'] for vraiIJ, score, nbCoups, dicoNN in lRes] for lRes in listeLRes])
listeLValeursJoueurs = [[] for _ in range(nbGenerations)]
listeLPoidsJoueurs = [[] for _ in range(nbGenerations)]
lIJ = sorted([vraiIJ for vraiIJ, *_ in listeLRes[0]])
for iG, lRes in enumerate(listeLRes):
    dico = {vraiIJ : (vraiIJ, score, nbCoups, dicoNN['0.weight']) for vraiIJ, score, nbCoups, dicoNN in lRes}
    listeLValeursJoueurs[iG] = [(iG, *dico[iJ][:3]) for iJ in lIJ]
    listeLPoidsJoueurs[iG] = [dico[iJ][3] for iJ in lIJ]
listeLValeursJoueurs = np.array(listeLValeursJoueurs)
listeLPoidsJoueurs = np.array(listeLPoidsJoueurs)

cmapPersonnel = LinearSegmentedColormap.from_list("mon_cmap", [(1,0,0), (0,0,1), (0,1,0)])

plt.style.use("dark_background")
fig, lAxes = plt.subplots(8,1)
fig.suptitle(f"{nbJoueurs} Joueurs || {nbBalles} Balles || {nbGenerations} nbGen || {nbParties} nbParties || {'⇝'.join(map(str, lCouches))} || {kNN} kNN")
lPositionsAxes = [[(0.1, 0.6, 0.8, 0.3), None],                        #graphique score
                  [(0.1, 0.2, 0.8, 0.3), None],                        #graphique nbCoups
                  [None, (0.1, 0.2, 0.8, 0.7)],                        #graphique dicoNN
                  [(0.55, 0.05, 0.2, 0.1), (0.8, 0.05, 0.15, 0.1)],    #boutonNormal
                  [(0.25, 0.05, 0.2, 0.1), (0.55, 0.05, 0.15, 0.1)],   #boutonAnimations
                  [None, (0.3, 0.05, 0.15, 0.1)],                      #slider v1
                  [None, (0.05, 0.05, 0.15, 0.1)],                     #slider v2
                  [(0.0, 0.95, 0.05, 0.05), (0.0, 0.95, 0.05, 0.05)]]  #BoutonMode
axe1, axe2, axe3, axe4, axe5, axe6, axe7, axe8 = lAxes

c1, c2 = 0.2, 0.4
color1, color2 = (c1,0,0), (c2,0,0)

normal = True
boutonNormal = Button(axe4, "None", color=color1, hovercolor=color2)
boutonNormal.on_clicked(changer_mode_normal)

animations = True
boutonAnimations = Button(axe5, "None", color=color1, hovercolor=color2)
boutonAnimations.on_clicked(changer_mode_animations)
animation1, animation2, animation3 = None, None, None

sliderV1 = Slider(axe6, "input 1", valmin=1, valmax=lCouches[0], valstep=1, valinit=1, track_color=color1, facecolor=color2)
sliderV1.on_changed(lambda _ : changer_valeur_slider(_, sliderV1))

sliderV2 = Slider(axe7, "input 2", valmin=1, valmax=lCouches[0], valstep=1, valinit=1, track_color=color1, facecolor=color2)
sliderV2.on_changed(lambda _ : changer_valeur_slider(_, sliderV2))

mode = 1
boutonMode = Button(axe8, "None", color=color1, hovercolor=color2)
boutonMode.on_clicked(changer_mode)

changer_mode_normal(None, False)
changer_mode_animations(None, False)
changer_valeur_slider(None, sliderV1, False)
changer_valeur_slider(None, sliderV2, False)
changer_mode(None, True)

plt.show()
