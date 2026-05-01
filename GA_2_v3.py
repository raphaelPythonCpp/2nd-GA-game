import pygame
from random import randint, uniform
import torch
from math import atan, tan, cos, sqrt, ceil
import os

class Jeu:
    def __init__(self, fenetre, police, horloge, nbJoueurs, lCouches, nbBalles, modeJeu, kNN, entrainementGA, chargementReseau):
        self.fenetre = fenetre
        self.wF, self.hF = self.fenetre.get_size()
        self.police = police
        self.horloge = horloge

        self.modeJeu = modeJeu
        self.kNN = kNN
        self.vMaxJoueurs, self.vMaxBalles = 10, 2.5
        self.lCouches = lCouches
        self.nbJoueurs = nbJoueurs
        self.lJoueurs = []
        for i in range(self.nbJoueurs):
            self.lJoueurs.append(Joueur(self, self.lCouches, vMax=self.vMaxJoueurs, kNN=self.kNN, iJ=i))
        if chargementReseau:
            with open(nomFichier, 'r') as fichier:
                listeLRes = eval(fichier.read())
            _, _, meilleurNN = listeLRes
            meilleurNN = {k : torch.tensor(v, dtype=torch.float32) for k, v in meilleurNN.items()}
            self.lJoueurs[0].NN.load_state_dict(meilleurNN)
        
        self.nbBalles = nbBalles
        self.lBalles = []
        for i in range(self.nbBalles):
            if self.modeJeu == 0:
                bord = 0
            elif self.modeJeu == 1:
                bord = i%2
            self.lBalles.append(Balle(self, vMax=self.vMaxBalles, bord=bord, xMin=self.lJoueurs[0].xMin-self.lJoueurs[0].r, xMax=self.lJoueurs[0].xMax+self.lJoueurs[0].r))

        if entrainementGA:
            lNNGA = self.GA(chargementReseau, 
                            nbJoueurs=int(input("GA : nbJoueurs = ")), nbJoueursExploration=int(input("GA : nbJoueursExploration : ")), nbBalles=self.nbBalles, 
                            nbGenerations=int(input("GA : nbGenerations = ")), nbParties=int(input("GA : nbParties = ")), 
                            nbCoupsMax1=int(input("GA : nbCoupsMax1 = ")), nbCoupsMax2=int(input("GA : nbCoupsMax2 = ")), tauxSurvivants=0.2,
                            vBallesMin=0.5*self.vMaxBalles, vBallesMax=2*self.vMaxBalles)
            for i, joueur in enumerate(self.lJoueurs[:min(self.nbJoueurs, len(lNNGA))]):
                joueur.NN.load_state_dict(lNNGA[i].state_dict())

    def reset(self, lJoueurs, lBalles):
        for joueur in lJoueurs:
            joueur.reset()
        nbBalles = len(lBalles)
        for iB, balle in enumerate(lBalles):
            if balle.bord == 0:
                yDebut = balle.lYDebut[balle.bord][0] - self.hF/nbBalles * iB
            elif balle.bord == 1:
                yDebut = balle.lYDebut[balle.bord][0] + self.hF/nbBalles * iB
            balle.reset(y=yDebut)
        return lJoueurs, lBalles

    def jouer(self, nbParties):
        nbCoupsMax = int(input("Jeu : nbCoupsMax = "))
        for iP in range(nbParties):
            self.lJoueurs, self.lBalles = self.reset(self.lJoueurs, self.lBalles)
            lJoueurs = self.lJoueurs.copy()
            if self.jouer_partie(lJoueurs, self.lBalles, nbCoupsMax):
                break

    def jouer_partie(self, lJoueurs, lBalles, nbCoupsMax, visuel=True, ralenti=True):
        quitterProgramme = False
        while lJoueurs and lJoueurs[0].nbCoups < nbCoupsMax:
            if ralenti:
                self.horloge.tick(100)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        lJoueurs = []
                        quitterProgramme = True
                        break
            lJoueurs = self.bouger(lJoueurs, lBalles)
            if visuel and lJoueurs:
                self.afficher(lJoueurs, lBalles)
        return quitterProgramme

    def bouger(self, lJoueurs, lBalles):
        for balle in lBalles :
            balle.bouger()
        for joueur in lJoueurs:
            joueur.jouer(lBalles)
        return [joueur for joueur in lJoueurs if joueur.enVie]

    def afficher(self, lJoueurs, lBalles):
        self.fenetre.fill((0,0,0))

        self.fenetre.blit(self.police.render(f"Score : {round(lJoueurs[0].score)} || nbCoups : {round(lJoueurs[0].nbCoups)} || d : {lJoueurs[0].d:.2f}", True, (255,100,100,100)), (0,0))
        for joueur in lJoueurs:
            joueur.afficher()
        for balle in lBalles:
            balle.afficher()
        
        pygame.display.flip()

    def GA(self, chargementReseau, nbJoueurs, nbJoueursExploration, nbBalles, nbGenerations, nbParties, nbCoupsMax1, nbCoupsMax2, tauxSurvivants, vBallesMin, vBallesMax):
        def generer_nn_mutation(NN, epsilon):
            dico2 = {}
            for k,v in NN.state_dict().items():
                bruit = torch.randn_like(v)
                dico2[k] = v + epsilon * bruit * (torch.abs(v) + 1)
                #dico2[k] = v + epsilon*bruit
            return dico2#{k : v + v*torch.randn_like(v)*epsilon + torch.randn_like(v)*epsilon for k,v in NN.state_dict()}

        def generer_nn_crossover(NN1, NN2, epsilon):
            dico2 = {}
            for (k1,v1),(k2,v2) in zip(NN1.state_dict().items(), NN2.state_dict().items()):
                if uniform(0,1) < 0.5:
                    k,v = k1, v1
                else :
                    k,v = k2, v2
                bruit = torch.randn_like(v)
                dico2[k] = v + epsilon * bruit * (torch.abs(v) + 1)
            return dico2

        def generation(iG, lJoueurs, lBalles, nbParties, nbCoupsMax1, nbCoupsMax2):
            lSumRes = [[joueur.iJ, iJ, 0, 0] for iJ, joueur in enumerate(lJoueurs)] #[i, score, nbCoups]
            for iP in range(nbParties):
                print('\r' + ' '*80 + '\r'f"GA : Gen {iG+1} ({int(100*(iG+1)/nbGenerations)}%) || Partie {iP+1} ({int(100*(iP+1)/nbParties)}%)", end='', flush=True)
                lJoueurs, lBalles = self.reset(lJoueurs, lBalles)
                if self.jouer_partie(lJoueurs, lBalles, nbCoupsMax=(nbCoupsMax1 if iP<nbParties/2 else nbCoupsMax2), visuel=False, ralenti=False):
                    break
                for iJ, joueur in enumerate(lJoueurs):
                    lSumRes[iJ][2] += joueur.score
                    lSumRes[iJ][3] += joueur.nbCoups
            
            lSumRes.sort(key=lambda triplet : (triplet[2]), reverse=True) #nbCoups
            lRes = [(vraiIJ, iJ, round(sommeScores/nbParties), round(sommeNbCoups/nbParties)) for vraiIJ, iJ, sommeScores, sommeNbCoups in lSumRes]
            lJoueurs = [lJoueurs[iJ] for vraiIJ, iJ, _, _ in lRes]
            print('\n', *[(score,nbCoups) for _, _, score, nbCoups in lRes])
            return lJoueurs, lRes

        torch.set_grad_enabled(False)
        #Exploration
        lJoueursExploration = []
        for i in range(nbJoueursExploration):
            lJoueursExploration.append(Joueur(self, self.lCouches, vMax=self.vMaxJoueurs, kNN=self.kNN, iJ=i))
        if chargementReseau:
            lJoueursExploration.NN.load_state_dict(self.lJoueurs[0].NN.state_dict())
        lBallesExploration = []
        for i in range(nbBalles):
            if self.modeJeu == 0:
                bord = 0
            elif self.modeJeu == 1:
                bord = i%2
            lBallesExploration.append(Balle(self, vMax=vBallesMin, bord=bord, xMin=lJoueursExploration[0].xMin-lJoueursExploration[0].r, xMax=lJoueursExploration[0].xMax+lJoueursExploration[0].r))
        lJoueursExploration, lBallesExploration = self.reset(lJoueursExploration, lBallesExploration)
        lJoueursExploration, _ = generation(-1, lJoueursExploration, lBallesExploration, nbParties, nbCoupsMax1, nbCoupsMax2)
        #GA
        nbSurvivants = max(1, round(nbJoueurs*tauxSurvivants))
        nbMutes = ceil((nbJoueurs-nbSurvivants)/nbSurvivants)
        lJoueurs = lJoueursExploration[:nbJoueurs]
        lBalles = []
        for i in range(nbBalles):
            if self.modeJeu == 0:
                bord = 0
            elif self.modeJeu == 1:
                bord = i%2
            lBalles.append(Balle(self, vMax=vBallesMin, bord=bord, xMin=lJoueurs[0].xMin-lJoueurs[0].r, xMax=lJoueurs[0].xMax+lJoueurs[0].r))
        lJoueurs, lBalles = self.reset(lJoueurs, lBalles)

        epsilon, epsilonFin = 0.15, 0.001
        epsilonDecrementation = (epsilonFin-epsilon) / nbGenerations

        listeLRes = [(nbJoueurs, nbBalles, nbGenerations, nbParties, self.lCouches, self.kNN), [], []] #[caracteristiques, listeLRes, meilleurNN]
        with open(nomFichier, 'w') as fichier:
            fichier.write(str(listeLRes))

        for iG in range(nbGenerations):
            for balle in lBalles:
                balle.vMax = vBallesMin + iG/(nbGenerations-1)*(vBallesMax-vBallesMin)
            lJoueurs, lRes = generation(iG, lJoueurs, lBalles, nbParties, nbCoupsMax1, nbCoupsMax2)
            with open(nomFichier, 'r') as fichier:
                listeLRes = eval(fichier.read())
            lRes = [(vraiIJ, score, nbCoups, {'0.weight' : joueur.NN.state_dict()['0.weight'][0].detach().cpu().numpy().tolist()}) for (vraiIJ, iJ,score,nbCoups),joueur in zip(lRes, lJoueurs)]
            listeLRes[1].append(lRes)
            listeLRes[2] = {k : v.detach().cpu().numpy().tolist() for k, v in lJoueurs[0].NN.state_dict().items()}
            with open(nomFichier, 'w') as fichier:
                fichier.write(str(listeLRes))
            for iS, survivant in enumerate(lJoueurs[:nbSurvivants]):
                survivant.reset()
                debut = min(nbJoueurs-1, nbSurvivants + nbMutes*iS)
                for iM, mute in enumerate(lJoueurs[debut : min(nbJoueurs, debut+nbMutes)]):
                    if uniform(0,1) < 0.5: #mutation
                        mute.NN.load_state_dict(generer_nn_mutation(survivant.NN, epsilon))
                    else :
                        survivant2 = lJoueurs[randint(0,nbSurvivants-1)]
                        mute.NN.load_state_dict(generer_nn_crossover(survivant.NN, survivant2.NN, epsilon))
                    mute.reset()
            epsilon -= epsilonDecrementation
        #torch.set_grad_enabled(True)
        
        return [joueur.NN for joueur in lJoueurs]



        




class Joueur:
    def __init__(self, jeu, lCouches, vMax, kNN, iJ):
        self.jeu = jeu

        self.iJ = iJ

        self.enVie = None
        self.score = 0
        self.kNN = kNN

        self.r = 15
        self.x, self.y = None, None
        self.xMin, self.xMax = self.jeu.wF/2-2*self.r, self.jeu.wF/2+2*self.r
        self.d = None
        if self.jeu.modeJeu == 0:
            self.yMin, self.yMax = self.jeu.hF - self.r, self.jeu.hF - self.r
        elif self.jeu.modeJeu == 1:
            self.yMin, self.yMax = self.jeu.hF / 2, self.jeu.hF / 2
        self.dx, self.dy = vMax, vMax
        self.couleur = (randint(128,255), randint(128,255), randint(128,255))

        lCouchesNN = []
        for i in range(1, len(lCouches)):
            lCouchesNN.append(torch.nn.Linear(lCouches[i-1], lCouches[i], bias=True))
            if i+1 < len(lCouches):
                lCouchesNN.append(torch.nn.LeakyReLU())
        self.NN = torch.nn.Sequential(*lCouchesNN)

        self.reset()

    def reset(self):
        self.enVie = True
        self.score, self.nbCoups = 0, 0
        self.x = self.jeu.wF/2 
        self.y = self.jeu.hF-self.r if self.jeu.modeJeu == 0 else self.jeu.hF/2 if self.jeu.modeJeu == 1 else None

    def jouer(self, lBalles):
        lMin = sorted([((balle.x-self.x)**2+(balle.y-self.y)**2, iB) for iB, balle in enumerate(lBalles)])[:self.kNN]
        balle = lBalles[lMin[0][1]]
        if sqrt(lMin[0][0]) < self.r+balle.r:
            self.enVie = False
            return
        lInput = [(self.x-self.jeu.wF/2)/(self.xMax-self.xMin)]
        for k in range(self.kNN):
            balle = lBalles[lMin[k][1]]
            dx, dy = (balle.x - self.x) / self.jeu.wF, (balle.y - self.y) / self.jeu.hF
            vx, vy = balle.dx / lBalles[0].vMax, balle.dy / lBalles[0].vMax
            lInput.extend([dx, dy, vx, vy])
        lInput = torch.tensor(lInput, dtype=torch.float32)

        lOutput = self.NN(lInput)
        """action = torch.argmax(lOutput).item()
        if action == 0: #rien
            self.x = self.x
        elif action == 1 : #gauche
            self.x = self.x-self.dx
        elif action == 2: #droite
            self.x = self.x+self.dx"""
        actionContinue = torch.clamp(lOutput[0], -1, 1).item()  # ∈ [-1, 1]
        self.x += actionContinue * self.dx
        self.x = min(self.xMax, max(self.xMin, self.x))
        self.y = min(self.yMax, max(self.yMin, self.y))

        self.d = max(1e-2, 2*abs((self.x-self.jeu.wF/2))/(self.xMax-self.xMin))
        self.score += 1 + 2.5 * (1-self.d)
        self.nbCoups += 1

    def afficher(self):
        pygame.draw.circle(self.jeu.fenetre, self.couleur, (self.x, self.y), self.r)





class Balle:
    def __init__(self, jeu, vMax, bord, xMin, xMax):
        self.jeu = jeu

        self.vMax = vMax
        self.r = 6
        self.x, self.y = None, None
        self.couleur = (randint(128,255), randint(128,255), randint(128,255))
        self.bord = bord
        self.lYDebut = [(-self.r, self.jeu.hF-self.r, self.jeu.hF+self.r), (self.jeu.hF+self.r, self.r, -self.r)] #yDebut, yFinTheorique, yFinReel
        self.xMin, self.xMax = xMin, xMax
        self.reset()

    def reset(self, y=None):
        self.x = uniform(self.xMin, self.xMax)
        if y is None:
            self.y = self.lYDebut[self.bord][0]
        else :
            self.y = y
        yFin = self.lYDebut[self.bord][1]
        xFin = uniform(self.xMin, self.xMax)
        dx, dy = xFin-self.x, yFin-self.y
        nbDeplacements = ceil(abs(dy) / self.vMax)
        self.dy = dy / nbDeplacements
        self.dx = dx / nbDeplacements

    def bouger(self):
        self.x += self.dx
        self.y += self.dy

        if (self.bord == 0 and self.y > self.lYDebut[self.bord][2]) or (self.bord == 1 and self.y < self.lYDebut[self.bord][2]):
            self.reset()

    def afficher(self): 
        pygame.draw.circle(self.jeu.fenetre, self.couleur, (self.x, self.y), self.r)


pygame.init()

wFenetre, hFenetre = 400, 500
fenetre = pygame.display.set_mode((wFenetre, hFenetre))
pygame.display.set_caption(f"{__file__[:-10]} Raphaël")

horloge = pygame.time.Clock()
police = pygame.font.SysFont("Arial", 10, bold=True, italic=False)

nomDossier = os.path.dirname(__file__)
nomFichier = os.path.join(nomDossier, "GA_2_v3.txt")
kNN = 2
env = Jeu(fenetre, police, horloge,
          nbJoueurs=10, lCouches=[1+4*kNN, 32, 16, 1], nbBalles=4,
          modeJeu=0, 
          kNN=kNN, entrainementGA=True, chargementReseau=False)

env.jouer(nbParties=100)

pygame.quit()